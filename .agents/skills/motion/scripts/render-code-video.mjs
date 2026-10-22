// Deterministic browser -> PNG frames -> FFmpeg. No video-generation service.
import {createServer} from 'node:http';
import {readFile,writeFile,mkdir,stat} from 'node:fs/promises';
import {createReadStream,existsSync} from 'node:fs';
import path from 'node:path';
import {createRequire} from 'node:module';
import {spawn} from 'node:child_process';
import {once} from 'node:events';
import {createHash} from 'node:crypto';

const args=process.argv.slice(2);
const opt=(key,fallback)=>{const n=args.indexOf('--'+key);return n<0?fallback:args[n+1]};
if(args.includes('--help')){
  console.log('node render-code-video.mjs --project DIR --mode video|stills|qa|preview --format landscape|vertical --out FILE_OR_DIR [--times 0,5,12] [--fps 60] [--audio FILE] [--ffmpeg EXE] [--browser msedge|chrome|chromium] [--playwright-root NODE_MODULES]');
  console.log('Page API: window.__motion {ready,duration,width,height,render(t),snapshot(): PNG data URL}. Default entry: index.html. Writes local files only. Preview binds to 127.0.0.1.');
  process.exit(0);
}
const root=path.resolve(opt('project','.')),mode=opt('mode','stills'),format=opt('format','landscape');
if(!['video','stills','qa','preview'].includes(mode))throw new Error('Unknown mode: '+mode);
if(!['landscape','vertical','square'].includes(format))throw new Error('Unknown format: '+format);
const fps=Number(opt('fps','60'));
if(!Number.isFinite(fps)||fps<1||fps>120)throw new Error('fps must be between 1 and 120');
async function bounded(promise,label,ms=30000){
  let timer;
  try{return await Promise.race([promise,new Promise((_,reject)=>timer=setTimeout(()=>reject(new Error(label+' timed out')),ms))])}
  finally{clearTimeout(timer)}
}
const mime={'.html':'text/html','.mjs':'text/javascript','.js':'text/javascript','.json':'application/json','.ttf':'font/ttf','.png':'image/png','.wav':'audio/wav','.mp4':'video/mp4','.svg':'image/svg+xml'};
const server=createServer(async(req,res)=>{
  try{
    const url=new URL(req.url,'http://127.0.0.1');
    const relative=decodeURIComponent(url.pathname==='/'?'/index.html':url.pathname);
    const file=path.resolve(root,'.'+relative);
    if(file!==root&&!file.startsWith(root+path.sep)){res.writeHead(403);res.end();return}
    const info=await stat(file);
    if(!info.isFile())throw new Error('not a file');
    res.writeHead(200,{'Content-Type':mime[path.extname(file)]||'application/octet-stream','Cache-Control':'no-store','Content-Length':info.size});
    createReadStream(file).pipe(res);
  }catch{res.writeHead(404);res.end('File not found')}
});
server.listen(Number(opt('port',mode==='preview'?'5273':'0')),'127.0.0.1');
await once(server,'listening');
const url=`http://127.0.0.1:${server.address().port}/?format=${format}${mode==='preview'?'':'&export=1'}`;
if(mode==='preview'){console.log('PREVIEW '+url);process.on('SIGINT',()=>server.close(()=>process.exit()));}
else {
  let browser,encoder;
  try{
    const packageRoot=opt('playwright-root');
    const require=createRequire(packageRoot?path.join(path.resolve(packageRoot),'__motion_loader__.cjs'):import.meta.url);
    const {chromium}=require('playwright-core');
    const channel=opt('browser','msedge');
    browser=await chromium.launch({...channel==='chromium'?{}:{channel},headless:true});
    const page=await browser.newPage({viewport:format==='vertical'?{width:1080,height:1920}:{width:1920,height:1080},deviceScaleFactor:1});
    const errors=[];
    page.on('pageerror',e=>errors.push(e.message));
    page.on('console',m=>{if(m.type()==='error')errors.push(m.text())});
    page.on('response',r=>{if(r.status()>=400)errors.push(`${r.status()} ${r.url()}`)});
    await page.goto(url,{waitUntil:'networkidle'});
    await page.waitForFunction(()=>window.__motion?.ready,{timeout:30000});
    const spec=await page.evaluate(()=>({duration:window.__motion.duration,width:window.__motion.width,height:window.__motion.height}));
    if(!Number.isFinite(spec.duration)||spec.duration<=0||!Number.isInteger(spec.width)||!Number.isInteger(spec.height)||spec.width<=0||spec.height<=0||spec.width%2||spec.height%2)throw new Error('Invalid duration or positive even pixel dimensions: '+JSON.stringify(spec));
    async function capture(t){
      const b64=await bounded(page.evaluate(async t=>{await window.__motion.render(t);return (await window.__motion.snapshot()).split(',')[1]},t),'Frame '+t);
      if(errors.length)throw new Error('Browser error: '+errors.join('\n'));
      const png=Buffer.from(b64,'base64');
      if(png.length<24||!png.subarray(0,8).equals(Buffer.from([137,80,78,71,13,10,26,10])))throw new Error('Snapshot must be a PNG');
      if(png.readUInt32BE(16)!==spec.width||png.readUInt32BE(20)!==spec.height)throw new Error('Snapshot dimensions do not match the declared video size');
      return png;
    }
    const out=path.resolve(opt('out',mode==='video'?'motion.mp4':'motion-stills'));
    if(mode==='stills'||mode==='qa'){
      await mkdir(out,{recursive:true});
      const times=String(opt('times',`0,${spec.duration*.25},${spec.duration*.5},${spec.duration*.75},${spec.duration-1/fps}`)).split(',').map(Number);
      if(times.some(t=>!Number.isFinite(t)||t<0||t>=spec.duration))throw new Error('Stills times must be finite and within the video duration');
      const results=[];
      for(const t of times){
        const a=await capture(t);await writeFile(path.join(out,`frame-${t.toFixed(3)}.png`),a);
        if(mode==='qa'){
          // Compare an ordered approach to t with a direct seek after another scene.
          for(const before of [Math.max(0,t-.1),Math.max(0,t-.05),t])await capture(before);
          const sequential=await capture(t);
          await capture(spec.duration-.05);await capture(0);
          const direct=await capture(t);
          const hash=b=>createHash('sha256').update(b).digest('hex');
          if(hash(a)!==hash(sequential)||hash(a)!==hash(direct))throw new Error('Non-deterministic frame at '+t);
          const pixels=await page.evaluate(async png=>{
            // Inspect the public snapshot, including SVG/DOM/WebGL implementations.
            const bitmap=await createImageBitmap(await (await fetch(png)).blob());
            const cv=new OffscreenCanvas(bitmap.width,bitmap.height),ctx=cv.getContext('2d');
            ctx.drawImage(bitmap,0,0);bitmap.close();
            const data=ctx.getImageData(0,0,cv.width,cv.height).data;
            let sum=0,sq=0,n=0;for(let i=0;i<data.length;i+=400){const l=(data[i]+data[i+1]+data[i+2])/3;sum+=l;sq+=l*l;n++}
            return {mean:sum/n,variance:sq/n-(sum/n)**2};
          },'data:image/png;base64,'+a.toString('base64'));
          if(pixels.variance<1)throw new Error('Flat/blank frame at '+t);
          results.push({time:t,sha256:hash(a),seekMatches:true,...pixels});
        }
      }
      if(mode==='qa')await writeFile(path.join(out,'qa.json'),JSON.stringify({format,fps,...spec,errors,frames:results},null,2));
      console.log(`${mode.toUpperCase()} ${times.length} frames | ${spec.width}x${spec.height} | ${spec.duration}s | ${out}`);
    }else{
      await mkdir(path.dirname(out),{recursive:true});
      if(existsSync(out)&&!args.includes('--overwrite'))throw new Error('Output exists; use a new path or --overwrite');
      const executable=opt('ffmpeg','ffmpeg'),audio=opt('audio');
      const ffargs=['-y','-hide_banner','-loglevel','error','-f','image2pipe','-framerate',String(fps),'-vcodec','png','-i','pipe:0'];
      if(audio)ffargs.push('-i',path.resolve(audio),'-map','0:v:0','-map','1:a:0','-c:a','aac','-b:a','256k');
      else ffargs.push('-an');
      ffargs.push('-c:v','libx264','-preset','medium','-crf',opt('crf','17'),'-pix_fmt','yuv420p','-color_primaries','bt709','-color_trc','bt709','-colorspace','bt709','-vf','scale=in_range=full:out_range=tv:out_color_matrix=bt709,format=yuv420p','-t',String(spec.duration),'-movflags','+faststart',out);
      encoder=spawn(executable,ffargs,{stdio:['pipe','ignore','pipe'],windowsHide:true});
      let encoderError,encoderLog='';
      encoder.on('error',e=>encoderError=e);encoder.stderr.on('data',d=>encoderLog+=d);encoder.stdin.on('error',e=>encoderError=e);
      const done=new Promise(resolve=>encoder.on('close',code=>resolve(code)));
      const drain=()=>bounded(new Promise((resolve,reject)=>{
        const cleanup=()=>{encoder.stdin.off('drain',ok);encoder.stdin.off('close',closed);encoder.stdin.off('error',failed);encoder.off('close',closed)};
        const ok=()=>{cleanup();resolve()};
        const failed=e=>{cleanup();reject(e)};
        const closed=()=>failed(new Error('FFmpeg closed before receiving all frames: '+encoderLog));
        encoder.stdin.once('drain',ok);encoder.stdin.once('close',closed);encoder.stdin.once('error',failed);encoder.once('close',closed);
        if(encoder.exitCode!==null||encoder.stdin.destroyed)closed();
      }),'FFmpeg backpressure');
      const frames=Math.round(spec.duration*fps);
      let lastReport=0;
      for(let i=0;i<frames;i++){
        if(encoderError||encoder.exitCode!==null)throw encoderError||new Error('FFmpeg stopped: '+encoderLog);
        const data=await capture(i/fps);
        if(!encoder.stdin.write(data))await drain();
        if(Date.now()-lastReport>5000){console.log(`RENDER ${i+1}/${frames} frames | ${format}`);lastReport=Date.now()}
      }
      encoder.stdin.end();const exit=await bounded(done,'FFmpeg completion',60000);
      if(exit!==0||encoderError)throw new Error(`FFmpeg failed (${exit}): ${encoderError||encoderLog}`);
      const check=spawn(executable,['-v','error','-i',out,'-f','null','-'],{stdio:['ignore','ignore','pipe'],windowsHide:true});
      let decodeLog='';check.stderr.on('data',d=>decodeLog+=d);const [decodeExit]=await once(check,'close');
      if(decodeExit!==0||decodeLog.trim())throw new Error('MP4 decode failed: '+decodeLog);
      await writeFile(out+'.json',JSON.stringify({format,fps,frames,...spec,audio:audio?path.basename(audio):null,decode:'passed',browserErrors:errors},null,2));
      console.log(`WROTE ${out} | ${frames} frames | ${spec.width}x${spec.height} | ${spec.duration}s | decode passed`);
    }
  }finally{encoder?.kill();await browser?.close();server.close()}
}
