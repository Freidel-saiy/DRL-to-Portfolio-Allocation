@echo off

:: To allow anaconda commands
set PATH_ANACONDA=D:\ANACONDA
call %PATH_ANACONDA%\Scripts\activate.bat %PATH_ANACONDA%

CD %~dp0
call activate mestrado
python "pre_RL.py"
pause
