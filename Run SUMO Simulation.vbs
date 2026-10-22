Option Explicit
Dim files, shell, root, python, runner
Set files = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")
root = files.GetParentFolderName(WScript.ScriptFullName)
python = root & "\.venv\Scripts\pythonw.exe"
runner = root & "\simulation\cooperative_loop_1km\run_simulation.py"
If Not files.FileExists(python) Then
    MsgBox "The project's Python environment is missing: " & python, 16, "Foundry SUMO"
    WScript.Quit 1
End If
shell.Run """" & python & """ -B """ & runner & """", 1, False
