library BetaFCN;

uses
    SysUtils,
    Classes,
    sfGamma in '..\..\Maths Functions\sfgamma.pas',
    AMath in '..\..\Maths Functions\amath.pas',
    sfBasic in '..\..\Maths Functions\sfbasic.pas',
    sfZeta in '..\..\Maths Functions\sfzeta.pas',
    sfExpInt in '..\..\Maths Functions\sfexpint.pas',
    sfHyperG in '..\..\Maths Functions\sfhyperg.pas',
    sfPoly in '..\..\Maths Functions\sfpoly.pas',
    sfEllInt in '..\..\Maths Functions\sfellint.pas',
    sfMisc in '..\..\Maths Functions\sfmisc.pas',
    sfBessel in '..\..\Maths Functions\sfbessel.pas',
    sfErf in '..\..\Maths Functions\sferf.pas';

type
    PArray=^TArray;
    TArray=array[1..100] of Double;

function Gamma(X:Double):Double;
begin
    Gamma:=sfc_gamma(X);
end;

procedure GetFunctionName(var Name:PAnsiChar); cdecl;
begin
    {The function name must
       - begin with the letter "f",
       - followed by a positive integer that uniquely identifies this function,
       - followed by a colon.
     The remainder of the string is optional (to describe the function)}
    StrCopy(Name, 'f3: Beta Distribution (f3=a3/(a5-a4)*Gamma(a1+a2)/(Gamma(a1)*Gamma(a2))*((x-a4)/(a5-a4))^(a1-1)*((a5-x)/(a5-a4))^(a2-1))');
end;

procedure GetFunctionValue(x,a:PArray; var y:Double); cdecl;
{Returns the value of the user-defined function in the y-variable based on the
input x array and a array, where a is the parameter array.}
var
    p1,p2,g1,g2:Double;
begin
    if (a[5]<=a[4]) or (x[1]<=a[4]) or (x[1]>=a[5]) then
        y:=0
    else
    begin
        p1:=Power((x[1]-a[4])/(a[5]-a[4]),a[1]-1);
        p2:=Power((a[5]-x[1])/(a[5]-a[4]),a[2]-1);
        g1:=Gamma(a[1]);
        g2:=Gamma(a[2]);
        if (g1=0) or (g1=PosInf_x) or (g2=0) or (g2=PosInf_x) then
            y:=0
        else
            y:=a[3]/(a[5]-a[4])*Gamma(a[1]+a[2])/(g1*g2)*p1*p2;
    end;
end;

procedure GetNumParameters(var NumParameters:Integer); cdecl;
begin
    {There are 5 parameters: a1, a2, a3, a4, a5}
    NumParameters:=5;
end;

procedure GetNumVariables(var NumVariables:Integer); cdecl;
begin
    {There is only 1 independent variable: x}
    NumVariables:=1;
end;

exports
    GetFunctionName index 1,
    GetFunctionValue index 2,
    GetNumParameters index 3,
    GetNumVariables index 4;

begin
end.
