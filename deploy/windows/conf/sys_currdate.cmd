 :: Works on any NT/2k machine independent of regional date settings
@ECHO off
SETLOCAL ENABLEEXTENSIONS
if "%date%A" LSS "A" (set toks=1-3) else (set toks=2-4)
for /f "tokens=2-4 delims=(-)" %%a in ('echo:^|date') do (
  for /f "tokens=%toks% delims=.-/ " %%i in ('date/t') do (
	set '%%a'=%%i
	set '%%b'=%%j
	set '%%c'=%%k))
if %'yy'% LSS 100 set 'yy'=20%'yy'%
set Today=%'yy'%-%'mm'%-%'dd'% 
ENDLOCAL & SET YEAR=%'yy'%& SET MONTH=%'mm'%& SET DAY=%'dd'%

:EOF