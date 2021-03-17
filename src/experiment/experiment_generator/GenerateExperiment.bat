echo off
echo "This will overwrite the existing experiment_template.py file!"
pause
for %%A in (*.xml) do XSLTProcessor.exe %%~fA experiment.xslt experiment.py

python experiment.py
xcopy %~dp0\experiment\setup_0\0.xml %~dp0\..\iterations\ /Y
del %~dp0\..\iterations\log.csv
pause