%
% Auxiliary file for input to the poincare program using the --auxsource option.
% Adds labels for the different $\beta$-values for the chirality.
%
% Copyright (C) 2025, Fredrik Jonsson, under GPLv3.
%
u:=1mm;
x:=-34u;
y:=24u;
dy:=5u;
greylevel=.5;
pair startpos, stoppos;

pickup pencircle scaled .6pt;

startpos:=(x,y-0dy);
stoppos:=(24.1u,2.0u);
dotlabel.lft(btex $\beta=2.0$ etex, startpos) withcolor black;
dotlabel(btex etex, startpos) withcolor greylevel[0,1];
dotlabel(btex etex, stoppos) withcolor greylevel[0,1];
draw startpos{dir -10}..{dir -10}stoppos withcolor greylevel[0,1];

startpos:=(x,y-1dy);
stoppos:=(-8.0u,-4.2u);
dotlabel.lft(btex $\beta=1.0$ etex, startpos) withcolor black;
dotlabel(btex etex, startpos) withcolor greylevel[0,1];
dotlabel(btex etex, stoppos) withcolor greylevel[0,1];
draw startpos{dir -15}..{dir -15}stoppos withcolor greylevel[0,1];

startpos:=(x,y-2dy);
stoppos:=(-22.5u,-3.4u);
dotlabel.lft(btex $\beta=0.0$ etex, startpos) withcolor black;
dotlabel(btex etex, startpos) withcolor greylevel[0,1];
dotlabel(btex etex, stoppos) withcolor greylevel[0,1];
draw startpos{dir -20}..{dir -20}stoppos withcolor greylevel[0,1];

startpos:=(x,y-3dy);
stoppos:=(-22.0u,2.0u);
dotlabel.lft(btex $\beta=-1.0$ etex, startpos) withcolor black;
dotlabel(btex etex, startpos) withcolor greylevel[0,1];
dotlabel(btex etex, stoppos) withcolor greylevel[0,1];
draw startpos{dir -5}..{dir -5}stoppos withcolor greylevel[0,1];

startpos:=(x,y-4dy);
stoppos:=(-9.2u,-20.0u);
dotlabel.lft(btex $\beta=-2.0$ etex, startpos) withcolor black;
dotlabel(btex etex, startpos) withcolor greylevel[0,1];
dotlabel(btex etex, stoppos) withcolor greylevel[0,1];
draw startpos{dir -30}..{dir 0}stoppos withcolor greylevel[0,1];

draw ((0pt,0pt)--(270pt,0pt)--(270pt,190pt)--(0pt,190pt)--(0pt,0pt))
  shifted (-142pt,-81pt) withcolor white;
