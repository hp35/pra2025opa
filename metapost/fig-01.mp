%%--------------------------------------------------------------------------
%% File:        fig-01.mp
%% Created:     20250405
%% Last change: 20250405
%% Description: MetaPost source-code for Fig. 1, showing a schematical
%%              picture of the setup of backward-wave optical parametric
%%              amplification in a quasi phase matching configuration.
%% Author:      Fredrik Jonsson <fredrik@jonsson.eu>
%%--------------------------------------------------------------------------

beginfig(1);
   path p;
   path pa, pb, pc, pd, pe, pf, pg, ph, pi, pj;
   path pa_prim, pb_prim, pc_prim, pd_prim, pe_prim,
        pf_prim, pg_prim, ph_prim, pi_prim, pj_prim;
   transform T;
   transform Ta, Th;

% %%
% %% Draw the shadings of the first mirror...
% %%
%    pickup pencircle scaled 0.6pt;
%    draw ((0.00mm,-1.25mm)--(1.25mm,-2.50mm)) withcolor 0.4 [white,black];
%    draw ((0.00mm, 0.00mm)--(2.50mm,-2.50mm)) withcolor 0.4 [white,black];
%    draw ((0.00mm, 1.25mm)--(3.75mm,-2.50mm)) withcolor 0.4 [white,black];
%    draw ((0.00mm, 2.50mm)--(5.00mm,-2.50mm)) withcolor 0.4 [white,black];
%    draw ((0.00mm, 3.75mm)--(5.00mm,-1.25mm)) withcolor 0.4 [white,black];
%    draw ((0.00mm, 5.00mm)--(5.00mm, 0.00mm)) withcolor 0.4 [white,black];
%    draw ((0.00mm, 6.25mm)--(5.00mm, 1.25mm)) withcolor 0.4 [white,black];
%    draw ((0.00mm, 7.50mm)--(5.00mm, 2.50mm)) withcolor 0.4 [white,black];
%    draw ((0.00mm, 8.75mm)--(5.00mm, 3.75mm)) withcolor 0.4 [white,black];
%    draw ((0.00mm,10.00mm)--(5.00mm, 5.00mm)) withcolor 0.4 [white,black];
%    draw ((0.00mm,11.25mm)--(5.00mm, 6.25mm)) withcolor 0.4 [white,black];
%    draw ((0.00mm,12.50mm)--(5.00mm, 7.50mm)) withcolor 0.4 [white,black];
%    draw ((0.00mm,13.75mm)--(5.00mm, 8.75mm)) withcolor 0.4 [white,black];
%    draw ((0.00mm,15.00mm)--(5.00mm,10.00mm)) withcolor 0.4 [white,black];
%    draw ((0.00mm,16.25mm)--(5.00mm,11.25mm)) withcolor 0.4 [white,black];
%    draw ((0.00mm,17.50mm)--(5.00mm,12.50mm)) withcolor 0.4 [white,black];
%    draw ((0.00mm,18.75mm)--(5.00mm,13.75mm)) withcolor 0.4 [white,black];
%    draw ((0.00mm,20.00mm)--(5.00mm,15.00mm)) withcolor 0.4 [white,black];
%    draw ((0.00mm,21.25mm)--(5.00mm,16.25mm)) withcolor 0.4 [white,black];
%    draw ((0.00mm,22.50mm)--(5.00mm,17.50mm)) withcolor 0.4 [white,black];
%    draw ((0.00mm,23.75mm)--(5.00mm,18.75mm)) withcolor 0.4 [white,black];
%    draw ((0.00mm,25.00mm)--(5.00mm,20.00mm)) withcolor 0.4 [white,black];
%    draw ((0.00mm,26.25mm)--(5.00mm,21.25mm)) withcolor 0.4 [white,black];
%    draw ((0.00mm,27.50mm)--(5.00mm,22.50mm)) withcolor 0.4 [white,black];
%    draw ((0.00mm,28.75mm)--(5.00mm,23.75mm)) withcolor 0.4 [white,black];
%    draw ((0.00mm,30.00mm)--(5.00mm,25.00mm)) withcolor 0.4 [white,black];
%    draw ((0.00mm,31.25mm)--(5.00mm,26.25mm)) withcolor 0.4 [white,black];
%    draw ((0.00mm,32.50mm)--(5.00mm,27.50mm)) withcolor 0.4 [white,black];
%    draw ((0.00mm,33.75mm)--(5.00mm,28.75mm)) withcolor 0.4 [white,black];
%    draw ((0.00mm,35.00mm)--(5.00mm,30.00mm)) withcolor 0.4 [white,black];
%    draw ((0.00mm,36.25mm)--(5.00mm,31.25mm)) withcolor 0.4 [white,black];
%    draw ((0.00mm,37.50mm)--(5.00mm,32.50mm)) withcolor 0.4 [white,black];
%    draw ((0.00mm,38.75mm)--(5.00mm,33.75mm)) withcolor 0.4 [white,black];
%    draw ((0.00mm,40.00mm)--(5.00mm,35.00mm)) withcolor 0.4 [white,black];
%    draw ((0.00mm,41.25mm)--(5.00mm,36.25mm)) withcolor 0.4 [white,black];
%    draw ((0.00mm,42.50mm)--(5.00mm,37.50mm)) withcolor 0.4 [white,black];
%    draw ((1.25mm,42.50mm)--(5.00mm,38.75mm)) withcolor 0.4 [white,black];
%    draw ((2.50mm,42.50mm)--(5.00mm,40.00mm)) withcolor 0.4 [white,black];
%    draw ((3.75mm,42.50mm)--(5.00mm,41.25mm)) withcolor 0.4 [white,black];
% 
% %%
% %% Draw the shadings of the second mirror...
% %%
%    pickup pencircle scaled 0.6pt;
%    draw ((45.00mm,-1.25mm)--(46.25mm, -2.50mm)) withcolor 0.4 [white,black];
%    draw ((45.00mm, 0.00mm)--(47.50mm, -2.50mm)) withcolor 0.4 [white,black];
%    draw ((45.00mm, 1.25mm)--(48.75mm, -2.50mm)) withcolor 0.4 [white,black];
%    draw ((45.00mm, 2.50mm)--(50.00mm, -2.50mm)) withcolor 0.4 [white,black];
%    draw ((45.00mm, 3.75mm)--(50.00mm, -1.25mm)) withcolor 0.4 [white,black];
%    draw ((45.00mm, 5.00mm)--(50.00mm, 0.00mm)) withcolor 0.4 [white,black];
%    draw ((45.00mm, 6.25mm)--(50.00mm, 1.25mm)) withcolor 0.4 [white,black];
%    draw ((45.00mm, 7.50mm)--(50.00mm, 2.50mm)) withcolor 0.4 [white,black];
%    draw ((45.00mm, 8.75mm)--(50.00mm, 3.75mm)) withcolor 0.4 [white,black];
%    draw ((45.00mm,10.00mm)--(50.00mm, 5.00mm)) withcolor 0.4 [white,black];
%    draw ((45.00mm,11.25mm)--(50.00mm, 6.25mm)) withcolor 0.4 [white,black];
%    draw ((45.00mm,12.50mm)--(50.00mm, 7.50mm)) withcolor 0.4 [white,black];
%    draw ((45.00mm,13.75mm)--(50.00mm, 8.75mm)) withcolor 0.4 [white,black];
%    draw ((45.00mm,15.00mm)--(50.00mm,10.00mm)) withcolor 0.4 [white,black];
%    draw ((45.00mm,16.25mm)--(50.00mm,11.25mm)) withcolor 0.4 [white,black];
%    draw ((45.00mm,17.50mm)--(50.00mm,12.50mm)) withcolor 0.4 [white,black];
%    draw ((45.00mm,18.75mm)--(50.00mm,13.75mm)) withcolor 0.4 [white,black];
%    draw ((45.00mm,20.00mm)--(50.00mm,15.00mm)) withcolor 0.4 [white,black];
%    draw ((45.00mm,21.25mm)--(50.00mm,16.25mm)) withcolor 0.4 [white,black];
%    draw ((45.00mm,22.50mm)--(50.00mm,17.50mm)) withcolor 0.4 [white,black];
%    draw ((45.00mm,23.75mm)--(50.00mm,18.75mm)) withcolor 0.4 [white,black];
%    draw ((45.00mm,25.00mm)--(50.00mm,20.00mm)) withcolor 0.4 [white,black];
%    draw ((45.00mm,26.25mm)--(50.00mm,21.25mm)) withcolor 0.4 [white,black];
%    draw ((45.00mm,27.50mm)--(50.00mm,22.50mm)) withcolor 0.4 [white,black];
%    draw ((45.00mm,28.75mm)--(50.00mm,23.75mm)) withcolor 0.4 [white,black];
%    draw ((45.00mm,30.00mm)--(50.00mm,25.00mm)) withcolor 0.4 [white,black];
%    draw ((45.00mm,31.25mm)--(50.00mm,26.25mm)) withcolor 0.4 [white,black];
%    draw ((45.00mm,32.50mm)--(50.00mm,27.50mm)) withcolor 0.4 [white,black];
%    draw ((45.00mm,33.75mm)--(50.00mm,28.75mm)) withcolor 0.4 [white,black];
%    draw ((45.00mm,35.00mm)--(50.00mm,30.00mm)) withcolor 0.4 [white,black];
%    draw ((45.00mm,36.25mm)--(50.00mm,31.25mm)) withcolor 0.4 [white,black];
%    draw ((45.00mm,37.50mm)--(50.00mm,32.50mm)) withcolor 0.4 [white,black];
%    draw ((45.00mm,38.75mm)--(50.00mm,33.75mm)) withcolor 0.4 [white,black];
%    draw ((45.00mm,40.00mm)--(50.00mm,35.00mm)) withcolor 0.4 [white,black];
%    draw ((45.00mm,41.25mm)--(50.00mm,36.25mm)) withcolor 0.4 [white,black];
%    draw ((45.00mm,42.50mm)--(50.00mm,37.50mm)) withcolor 0.4 [white,black];
%    draw ((46.25mm,42.50mm)--(50.00mm,38.75mm)) withcolor 0.4 [white,black];
%    draw ((47.50mm,42.50mm)--(50.00mm,40.00mm)) withcolor 0.4 [white,black];
%    draw ((48.25mm,42.50mm)--(50.00mm,41.25mm)) withcolor 0.4 [white,black];
  
%%
%% Draw the crystal.
%%
   pickup pencircle scaled 1.0pt;
   draw ((0mm,-2.5mm)--(50mm,-2.5mm)--(50mm,34.5mm)
         --(0mm,34.5mm)--(0mm,-2.5mm));
   pickup pencircle scaled 0.6pt;
   for x = 4mm step 4mm until 20mm:
     draw ((x,-2.5mm)--(x,34.5mm)) withcolor black;
   endfor
   for x = 30mm step 4mm until 50mm:
     draw ((x,-2.5mm)--(x,34.5mm)) withcolor black;
   endfor
   for x = 2mm step 8mm until 20mm:
     label(btex $+$ etex,(x,16mm));
   endfor
   for x = 32mm step 8mm until 50mm:
     label(btex $+$ etex,(x,16mm));
   endfor
   for x = 6mm step 8mm until 20mm:
     label(btex $-$ etex,(x,16mm));
   endfor
   for x = 36mm step 8mm until 50mm:
     label(btex $-$ etex,(x,16mm));
   endfor
   label(btex $\cdots$ etex,(25mm,16mm));

%%
%% Draw the z coordinate axis below the setup...
%%
   pickup pencircle scaled 0.6pt;
   drawarrow ((-5mm,-7.5mm)--(60mm,-7.5mm));
   draw ((0mm,-8.5mm)--(0mm,-6.5mm)) withcolor 1.0 [white,black];
   label.ulft(btex $0$ etex, (0mm,-7.0mm));
   draw ((50mm,-8.5mm)--(50mm,-6.5mm)) withcolor 1.0 [white,black];
   label.urt(btex $L$ etex, (50mm,-7.0mm));
   label.top(btex $z$ etex, (60mm,-6.5mm));

%%
%% Draw the forward and backward propagating optical LCP/RCP
%% fields inside and outside the cavity....
%%
   whitegap := 3.0pt;
   ellipsethickness := 0.6pt;
   arrowthickness := 1.0pt;
   pa := makepath makepen
            (-1.3357 ,-0.8090)..(-1.3700 ,-0.7604)..(-1.4006 ,-0.7071)..
            (-1.4270 ,-0.6494)..(-1.4490 ,-0.5878)..(-1.4662 ,-0.5225)..
            (-1.4784 ,-0.4540)..(-1.4854 ,-0.3827)..(-1.4870 ,-0.3090)..
            (-1.4830 ,-0.2334)..(-1.4733 ,-0.1564)..(-1.4578 ,-0.0785)..
            (-1.4365 , 0.0000)..(-1.4093 , 0.0785)..(-1.3762 , 0.1564)..
            (-1.3374 , 0.2334)..(-1.2928 , 0.3090)..(-1.2427 , 0.3827)..
            (-1.1872 , 0.4540)..(-1.1264 , 0.5225)..(-1.0607 , 0.5878)..
            (-0.9902 , 0.6494)..(-0.9152 , 0.7071)..(-0.8361 , 0.7604)..
            (-0.7532 , 0.8090)..(-0.6668 , 0.8526)..(-0.5774 , 0.8910)..
            (-0.4853 , 0.9239)..(-0.3910 , 0.9511)..(-0.2948 , 0.9724)..
            (-0.1973 , 0.9877)..(-0.0989 , 0.9969)..(      0 , 1.0000);

   pa_prim := makepath makepen
            (-1.3357 ,-0.8090)..(-1.3700 ,-0.7604)..(-1.4006 ,-0.7071)..
            (-1.4270 ,-0.6494)..(-1.4490 ,-0.5878)..(-1.4662 ,-0.5225)..
            (-1.4784 ,-0.4540)..(-1.4854 ,-0.3827)..(-1.4870 ,-0.3090)..
            (-1.4830 ,-0.2334)..(-1.4733 ,-0.1564)..(-1.4578 ,-0.0785)..
            (-1.4365 , 0.0000)..(-1.4093 , 0.0785)..(-1.3762 , 0.1564)..
            (-1.3374 , 0.2334)..(-1.2928 , 0.3090)..(-1.2427 , 0.3827)..
            (-1.1872 , 0.4540)..(-1.1264 , 0.5225)..(-1.0607 , 0.5878)..
            (-0.9902 , 0.6494)..(-0.9152 , 0.7071)..(-0.8361 , 0.7604)..
            (-0.7532 , 0.8090)..(-0.6668 , 0.8526)..(-0.5774 , 0.8910)..
            (-0.4853 , 0.9239)..(-0.3910 , 0.9511)..(-0.2948 , 0.9724)..
            (-0.1973 , 0.9877)..(-0.0989 , 0.9969)..(      0 , 1.0000)..
            ( 0.0989 , 0.9969)..( 0.1973 , 0.9877)..( 0.2948 , 0.9724);

   pb := makepath makepen
            (      0 , 1.0000)..( 0.0989 , 0.9969)..( 0.1973 , 0.9877)..
            ( 0.2948 , 0.9724)..( 0.3910 , 0.9511)..( 0.4853 , 0.9239)..
            ( 0.5774 , 0.8910)..( 0.6668 , 0.8526)..( 0.7532 , 0.8090)..
            ( 0.8361 , 0.7604)..( 0.9152 , 0.7071)..( 0.9902 , 0.6494)..
            ( 1.0607 , 0.5878)..( 1.1264 , 0.5225)..( 1.1872 , 0.4540)..
            ( 1.2427 , 0.3827)..( 1.2928 , 0.3090)..( 1.3374 , 0.2334)..
            ( 1.3762 , 0.1564)..( 1.4093 , 0.0785)..( 1.4365 , 0.0000)..
            ( 1.4578 ,-0.0785)..( 1.4733 ,-0.1564)..( 1.4830 ,-0.2334)..
            ( 1.4870 ,-0.3090)..( 1.4854 ,-0.3827)..( 1.4784 ,-0.4540)..
            ( 1.4662 ,-0.5225)..( 1.4490 ,-0.5878)..( 1.4270 ,-0.6494)..
            ( 1.4006 ,-0.7071)..( 1.3700 ,-0.7604)..( 1.3357 ,-0.8090)..
            ( 1.2978 ,-0.8526)..( 1.2570 ,-0.8910)..( 1.2134 ,-0.9239)..
            ( 1.1676 ,-0.9511)..( 1.1200 ,-0.9724)..( 1.0710 ,-0.9877)..
            ( 1.0212 ,-0.9969)..
            ( 0.9708 ,-1.0000);

   pb_prim := makepath makepen
            (-0.2948 , 0.9724)..(-0.1973 , 0.9877)..(-0.0989 , 0.9969)..
            (      0 , 1.0000)..( 0.0989 , 0.9969)..( 0.1973 , 0.9877)..
            ( 0.2948 , 0.9724)..( 0.3910 , 0.9511)..( 0.4853 , 0.9239)..
            ( 0.5774 , 0.8910)..( 0.6668 , 0.8526)..( 0.7532 , 0.8090)..
            ( 0.8361 , 0.7604)..( 0.9152 , 0.7071)..( 0.9902 , 0.6494)..
            ( 1.0607 , 0.5878)..( 1.1264 , 0.5225)..( 1.1872 , 0.4540)..
            ( 1.2427 , 0.3827)..( 1.2928 , 0.3090)..( 1.3374 , 0.2334)..
            ( 1.3762 , 0.1564)..( 1.4093 , 0.0785)..( 1.4365 , 0.0000)..
            ( 1.4578 ,-0.0785)..( 1.4733 ,-0.1564)..( 1.4830 ,-0.2334)..
            ( 1.4870 ,-0.3090)..( 1.4854 ,-0.3827)..( 1.4784 ,-0.4540)..
            ( 1.4662 ,-0.5225)..( 1.4490 ,-0.5878)..( 1.4270 ,-0.6494)..
            ( 1.4006 ,-0.7071)..( 1.3700 ,-0.7604)..( 1.3357 ,-0.8090)..
            ( 1.2978 ,-0.8526)..( 1.2570 ,-0.8910)..( 1.2134 ,-0.9239)..
            ( 1.1676 ,-0.9511)..( 1.1200 ,-0.9724)..( 1.0710 ,-0.9877)..
            ( 1.0212 ,-0.9969)..( 0.9708 ,-1.0000)..( 0.9205 ,-0.9969)..
            ( 0.8706 ,-0.9877)..( 0.8216 ,-0.9724);

   pc := makepath makepen
            ( 0.9708 ,-1.0000)..( 0.9205 ,-0.9969)..( 0.8706 ,-0.9877)..
            ( 0.8216 ,-0.9724)..( 0.7740 ,-0.9511)..( 0.7282 ,-0.9239)..
            ( 0.6847 ,-0.8910)..( 0.6438 ,-0.8526)..( 0.6059 ,-0.8090)..
            ( 0.5716 ,-0.7604)..( 0.5410 ,-0.7071)..( 0.5146 ,-0.6494)..
            ( 0.4926 ,-0.5878)..( 0.4754 ,-0.5225)..( 0.4632 ,-0.4540)..
            ( 0.4562 ,-0.3827)..( 0.4546 ,-0.3090)..( 0.4586 ,-0.2334)..
            ( 0.4683 ,-0.1564)..( 0.4838 ,-0.0785)..( 0.5052 ,-0.0000)..
            ( 0.5324 , 0.0785)..( 0.5654 , 0.1564)..( 0.6042 , 0.2334)..
            ( 0.6488 , 0.3090)..( 0.6989 , 0.3827)..( 0.7544 , 0.4540)..
            ( 0.8152 , 0.5225)..( 0.8809 , 0.5878)..( 0.9515 , 0.6494)..
            ( 1.0264 , 0.7071)..( 1.1055 , 0.7604)..( 1.1884 , 0.8090)..
            ( 1.2748 , 0.8526)..( 1.3642 , 0.8910)..( 1.4563 , 0.9239)..
            ( 1.5506 , 0.9511)..( 1.6468 , 0.9724)..( 1.7443 , 0.9877)..
            ( 1.8427 , 0.9969)..( 1.9416 , 1.0000);

   pc_prim := makepath makepen
            ( 1.1200 ,-0.9724)..( 1.0710 ,-0.9877)..( 1.0212 ,-0.9969)..
            ( 0.9708 ,-1.0000)..( 0.9205 ,-0.9969)..( 0.8706 ,-0.9877)..
            ( 0.8216 ,-0.9724)..( 0.7740 ,-0.9511)..( 0.7282 ,-0.9239)..
            ( 0.6847 ,-0.8910)..( 0.6438 ,-0.8526)..( 0.6059 ,-0.8090)..
            ( 0.5716 ,-0.7604)..( 0.5410 ,-0.7071)..( 0.5146 ,-0.6494)..
            ( 0.4926 ,-0.5878)..( 0.4754 ,-0.5225)..( 0.4632 ,-0.4540)..
            ( 0.4562 ,-0.3827)..( 0.4546 ,-0.3090)..( 0.4586 ,-0.2334)..
            ( 0.4683 ,-0.1564)..( 0.4838 ,-0.0785)..( 0.5052 ,-0.0000)..
            ( 0.5324 , 0.0785)..( 0.5654 , 0.1564)..( 0.6042 , 0.2334)..
            ( 0.6488 , 0.3090)..( 0.6989 , 0.3827)..( 0.7544 , 0.4540)..
            ( 0.8152 , 0.5225)..( 0.8809 , 0.5878)..( 0.9515 , 0.6494)..
            ( 1.0264 , 0.7071)..( 1.1055 , 0.7604)..( 1.1884 , 0.8090)..
            ( 1.2748 , 0.8526)..( 1.3642 , 0.8910)..( 1.4563 , 0.9239)..
            ( 1.5506 , 0.9511)..( 1.6468 , 0.9724)..( 1.7443 , 0.9877)..
            ( 1.8427 , 0.9969)..( 1.9416 , 1.0000)..( 2.0405 , 0.9969)..
            ( 2.1389 , 0.9877)..( 2.2364 , 0.9724);


   pd := makepath makepen
            ( 1.9416 , 1.0000)..( 2.0405 , 0.9969)..( 2.1389 , 0.9877)..
            ( 2.2364 , 0.9724)..( 2.3326 , 0.9511)..( 2.4269 , 0.9239)..
            ( 2.5190 , 0.8910)..( 2.6084 , 0.8526)..( 2.6948 , 0.8090)..
            ( 2.7777 , 0.7604)..( 2.8568 , 0.7071)..( 2.9318 , 0.6494)..
            ( 3.0023 , 0.5878)..( 3.0680 , 0.5225)..( 3.1288 , 0.4540)..
            ( 3.1843 , 0.3827)..( 3.2344 , 0.3090)..( 3.2790 , 0.2334)..
            ( 3.3178 , 0.1564)..( 3.3509 , 0.0785)..( 3.3781 , 0.0000)..
            ( 3.3994 ,-0.0785)..( 3.4149 ,-0.1564)..( 3.4246 ,-0.2334)..
            ( 3.4286 ,-0.3090)..( 3.4270 ,-0.3827)..( 3.4200 ,-0.4540)..
            ( 3.4078 ,-0.5225)..( 3.3906 ,-0.5878)..( 3.3686 ,-0.6494)..
            ( 3.3422 ,-0.7071)..( 3.3116 ,-0.7604)..( 3.2773 ,-0.8090)..
            ( 3.2395 ,-0.8526)..( 3.1986 ,-0.8910)..( 3.1550 ,-0.9239)..
            ( 3.1092 ,-0.9511)..( 3.0616 ,-0.9724)..( 3.0127 ,-0.9877)..
            ( 2.9628 ,-0.9969)..( 2.9124 ,-1.0000);

   pd_prim := makepath makepen
            ( 1.6468 , 0.9724)..( 1.7443 , 0.9877)..( 1.8427 , 0.9969)..
            ( 1.9416 , 1.0000)..( 2.0405 , 0.9969)..( 2.1389 , 0.9877)..
            ( 2.2364 , 0.9724)..( 2.3326 , 0.9511)..( 2.4269 , 0.9239)..
            ( 2.5190 , 0.8910)..( 2.6084 , 0.8526)..( 2.6948 , 0.8090)..
            ( 2.7777 , 0.7604)..( 2.8568 , 0.7071)..( 2.9318 , 0.6494)..
            ( 3.0023 , 0.5878)..( 3.0680 , 0.5225)..( 3.1288 , 0.4540)..
            ( 3.1843 , 0.3827)..( 3.2344 , 0.3090)..( 3.2790 , 0.2334)..
            ( 3.3178 , 0.1564)..( 3.3509 , 0.0785)..( 3.3781 , 0.0000)..
            ( 3.3994 ,-0.0785)..( 3.4149 ,-0.1564)..( 3.4246 ,-0.2334)..
            ( 3.4286 ,-0.3090)..( 3.4270 ,-0.3827)..( 3.4200 ,-0.4540)..
            ( 3.4078 ,-0.5225)..( 3.3906 ,-0.5878)..( 3.3686 ,-0.6494)..
            ( 3.3422 ,-0.7071)..( 3.3116 ,-0.7604)..( 3.2773 ,-0.8090)..
            ( 3.2395 ,-0.8526)..( 3.1986 ,-0.8910)..( 3.1550 ,-0.9239)..
            ( 3.1092 ,-0.9511)..( 3.0616 ,-0.9724)..( 3.0127 ,-0.9877)..
            ( 2.9628 ,-0.9969)..( 2.9124 ,-1.0000)..( 2.8621 ,-0.9969)..
            ( 2.8122 ,-0.9877)..( 2.7632 ,-0.9724);

   pe := makepath makepen
            ( 2.9124 ,-1.0000)..( 2.8621 ,-0.9969)..( 2.8122 ,-0.9877)..
            ( 2.7632 ,-0.9724)..( 2.7156 ,-0.9511)..( 2.6698 ,-0.9239)..
            ( 2.6263 ,-0.8910)..( 2.5854 ,-0.8526)..( 2.5476 ,-0.8090)..
            ( 2.5132 ,-0.7604)..( 2.4826 ,-0.7071)..( 2.4562 ,-0.6494)..
            ( 2.4342 ,-0.5878)..( 2.4170 ,-0.5225)..( 2.4048 ,-0.4540)..
            ( 2.3978 ,-0.3827)..( 2.3962 ,-0.3090)..( 2.4002 ,-0.2334)..
            ( 2.4099 ,-0.1564)..( 2.4254 ,-0.0785)..( 2.4468 ,-0.0000)..
            ( 2.4740 , 0.0785)..( 2.5070 , 0.1564)..( 2.5459 , 0.2334)..
            ( 2.5904 , 0.3090)..( 2.6405 , 0.3827)..( 2.6960 , 0.4540)..
            ( 2.7568 , 0.5225)..( 2.8226 , 0.5878)..( 2.8931 , 0.6494)..
            ( 2.9680 , 0.7071)..( 3.0471 , 0.7604)..( 3.1300 , 0.8090)..
            ( 3.2164 , 0.8526)..( 3.3058 , 0.8910)..( 3.3979 , 0.9239)..
            ( 3.4922 , 0.9511)..( 3.5884 , 0.9724)..( 3.6859 , 0.9877)..
            ( 3.7843 , 0.9969)..( 3.8832 , 1.0000);

   pe_prim := makepath makepen
            ( 3.0616 ,-0.9724)..( 3.0127 ,-0.9877)..( 2.9628 ,-0.9969)..
            ( 2.9124 ,-1.0000)..( 2.8621 ,-0.9969)..( 2.8122 ,-0.9877)..
            ( 2.7632 ,-0.9724)..( 2.7156 ,-0.9511)..( 2.6698 ,-0.9239)..
            ( 2.6263 ,-0.8910)..( 2.5854 ,-0.8526)..( 2.5476 ,-0.8090)..
            ( 2.5132 ,-0.7604)..( 2.4826 ,-0.7071)..( 2.4562 ,-0.6494)..
            ( 2.4342 ,-0.5878)..( 2.4170 ,-0.5225)..( 2.4048 ,-0.4540)..
            ( 2.3978 ,-0.3827)..( 2.3962 ,-0.3090)..( 2.4002 ,-0.2334)..
            ( 2.4099 ,-0.1564)..( 2.4254 ,-0.0785)..( 2.4468 ,-0.0000)..
            ( 2.4740 , 0.0785)..( 2.5070 , 0.1564)..( 2.5459 , 0.2334)..
            ( 2.5904 , 0.3090)..( 2.6405 , 0.3827)..( 2.6960 , 0.4540)..
            ( 2.7568 , 0.5225)..( 2.8226 , 0.5878)..( 2.8931 , 0.6494)..
            ( 2.9680 , 0.7071)..( 3.0471 , 0.7604)..( 3.1300 , 0.8090)..
            ( 3.2164 , 0.8526)..( 3.3058 , 0.8910)..( 3.3979 , 0.9239)..
            ( 3.4922 , 0.9511)..( 3.5884 , 0.9724)..( 3.6859 , 0.9877)..
            ( 3.7843 , 0.9969)..( 3.8832 , 1.0000)..( 3.9821 , 0.9969)..
            ( 4.0805 , 0.9877)..( 4.1781 , 0.9724);

   pf := makepath makepen
            ( 3.8832 , 1.0000)..( 3.9821 , 0.9969)..( 4.0805 , 0.9877)..
            ( 4.1781 , 0.9724)..( 4.2742 , 0.9511)..( 4.3685 , 0.9239)..
            ( 4.4606 , 0.8910)..( 4.5500 , 0.8526)..( 4.6364 , 0.8090)..
            ( 4.7193 , 0.7604)..( 4.7984 , 0.7071)..( 4.8734 , 0.6494)..
            ( 4.9439 , 0.5878)..( 5.0096 , 0.5225)..( 5.0704 , 0.4540)..
            ( 5.1259 , 0.3827)..( 5.1761 , 0.3090)..( 5.2206 , 0.2334)..
            ( 5.2594 , 0.1564)..( 5.2925 , 0.0785)..( 5.3197 , 0.0000)..
            ( 5.3410 ,-0.0785)..( 5.3565 ,-0.1564)..( 5.3662 ,-0.2334)..
            ( 5.3702 ,-0.3090)..( 5.3686 ,-0.3827)..( 5.3616 ,-0.4540)..
            ( 5.3494 ,-0.5225)..( 5.3322 ,-0.5878)..( 5.3102 ,-0.6494)..
            ( 5.2838 ,-0.7071)..( 5.2533 ,-0.7604)..( 5.2189 ,-0.8090)..
            ( 5.1811 ,-0.8526)..( 5.1402 ,-0.8910)..( 5.0966 ,-0.9239)..
            ( 5.0508 ,-0.9511)..( 5.0032 ,-0.9724)..( 4.9543 ,-0.9877)..
            ( 4.9044 ,-0.9969)..( 4.8540 ,-1.0000);

   pf_prim := makepath makepen
            ( 3.5884 , 0.9724)..( 3.6859 , 0.9877)..( 3.7843 , 0.9969)..
            ( 3.8832 , 1.0000)..( 3.9821 , 0.9969)..( 4.0805 , 0.9877)..
            ( 4.1781 , 0.9724)..( 4.2742 , 0.9511)..( 4.3685 , 0.9239)..
            ( 4.4606 , 0.8910)..( 4.5500 , 0.8526)..( 4.6364 , 0.8090)..
            ( 4.7193 , 0.7604)..( 4.7984 , 0.7071)..( 4.8734 , 0.6494)..
            ( 4.9439 , 0.5878)..( 5.0096 , 0.5225)..( 5.0704 , 0.4540)..
            ( 5.1259 , 0.3827)..( 5.1761 , 0.3090)..( 5.2206 , 0.2334)..
            ( 5.2594 , 0.1564)..( 5.2925 , 0.0785)..( 5.3197 , 0.0000)..
            ( 5.3410 ,-0.0785)..( 5.3565 ,-0.1564)..( 5.3662 ,-0.2334)..
            ( 5.3702 ,-0.3090)..( 5.3686 ,-0.3827)..( 5.3616 ,-0.4540)..
            ( 5.3494 ,-0.5225)..( 5.3322 ,-0.5878)..( 5.3102 ,-0.6494)..
            ( 5.2838 ,-0.7071)..( 5.2533 ,-0.7604)..( 5.2189 ,-0.8090)..
            ( 5.1811 ,-0.8526)..( 5.1402 ,-0.8910)..( 5.0966 ,-0.9239)..
            ( 5.0508 ,-0.9511)..( 5.0032 ,-0.9724)..( 4.9543 ,-0.9877)..
            ( 4.9044 ,-0.9969)..( 4.8540 ,-1.0000)..( 4.8037 ,-0.9969)..
            ( 4.7538 ,-0.9877)..( 4.7048 ,-0.9724);

   pg := makepath makepen
            ( 4.8540 ,-1.0000)..( 4.8037 ,-0.9969)..( 4.7538 ,-0.9877)..
            ( 4.7048 ,-0.9724)..( 4.6572 ,-0.9511)..( 4.6114 ,-0.9239)..
            ( 4.5679 ,-0.8910)..( 4.5270 ,-0.8526)..( 4.4892 ,-0.8090)..
            ( 4.4548 ,-0.7604)..( 4.4242 ,-0.7071)..( 4.3978 ,-0.6494)..
            ( 4.3758 ,-0.5878)..( 4.3586 ,-0.5225)..( 4.3464 ,-0.4540)..
            ( 4.3394 ,-0.3827)..( 4.3378 ,-0.3090)..( 4.3418 ,-0.2334)..
            ( 4.3515 ,-0.1564)..( 4.3670 ,-0.0785)..( 4.3884 ,-0.0000)..
            ( 4.4156 , 0.0785)..( 4.4486 , 0.1564)..( 4.4875 , 0.2334)..
            ( 4.5320 , 0.3090)..( 4.5821 , 0.3827)..( 4.6377 , 0.4540)..
            ( 4.6984 , 0.5225)..( 4.7642 , 0.5878)..( 4.8347 , 0.6494)..
            ( 4.9096 , 0.7071)..( 4.9887 , 0.7604)..( 5.0717 , 0.8090)..
            ( 5.1580 , 0.8526)..( 5.2474 , 0.8910)..( 5.3395 , 0.9239)..
            ( 5.4339 , 0.9511)..( 5.5300 , 0.9724)..( 5.6275 , 0.9877)..
            ( 5.7259 , 0.9969)..( 5.8248 , 1.0000);

   pg_prim := makepath makepen
            ( 5.0032 ,-0.9724)..( 4.9543 ,-0.9877)..( 4.9044 ,-0.9969)..
            ( 4.8540 ,-1.0000)..( 4.8037 ,-0.9969)..( 4.7538 ,-0.9877)..
            ( 4.7048 ,-0.9724)..( 4.6572 ,-0.9511)..( 4.6114 ,-0.9239)..
            ( 4.5679 ,-0.8910)..( 4.5270 ,-0.8526)..( 4.4892 ,-0.8090)..
            ( 4.4548 ,-0.7604)..( 4.4242 ,-0.7071)..( 4.3978 ,-0.6494)..
            ( 4.3758 ,-0.5878)..( 4.3586 ,-0.5225)..( 4.3464 ,-0.4540)..
            ( 4.3394 ,-0.3827)..( 4.3378 ,-0.3090)..( 4.3418 ,-0.2334)..
            ( 4.3515 ,-0.1564)..( 4.3670 ,-0.0785)..( 4.3884 ,-0.0000)..
            ( 4.4156 , 0.0785)..( 4.4486 , 0.1564)..( 4.4875 , 0.2334)..
            ( 4.5320 , 0.3090)..( 4.5821 , 0.3827)..( 4.6377 , 0.4540)..
            ( 4.6984 , 0.5225)..( 4.7642 , 0.5878)..( 4.8347 , 0.6494)..
            ( 4.9096 , 0.7071)..( 4.9887 , 0.7604)..( 5.0717 , 0.8090)..
            ( 5.1580 , 0.8526)..( 5.2474 , 0.8910)..( 5.3395 , 0.9239)..
            ( 5.4339 , 0.9511)..( 5.5300 , 0.9724)..( 5.6275 , 0.9877)..
            ( 5.7259 , 0.9969)..( 5.8248 , 1.0000)..( 5.9237 , 0.9969)..
            ( 6.0222 , 0.9877)..( 6.1197 , 0.9724);

   ph := makepath makepen
            ( 5.8248 , 1.0000)..( 5.9237 , 0.9969)..( 6.0222 , 0.9877)..
            ( 6.1197 , 0.9724)..( 6.2158 , 0.9511)..( 6.3101 , 0.9239)..
            ( 6.4022 , 0.8910)..( 6.4916 , 0.8526)..( 6.5780 , 0.8090)..
            ( 6.6609 , 0.7604)..( 6.7400 , 0.7071)..( 6.8150 , 0.6494)..
            ( 6.8855 , 0.5878)..( 6.9513 , 0.5225)..( 7.0120 , 0.4540)..
            ( 7.0675 , 0.3827)..( 7.1177 , 0.3090)..( 7.1622 , 0.2334)..
            ( 7.2010 , 0.1564)..( 7.2341 , 0.0785)..( 7.2613 ,-0.0000)..
            ( 7.2826 ,-0.0785)..( 7.2981 ,-0.1564)..( 7.3078 ,-0.2334)..
            ( 7.3118 ,-0.3090)..( 7.3102 ,-0.3827)..( 7.3033 ,-0.4540)..
            ( 7.2910 ,-0.5225)..( 7.2738 ,-0.5878)..( 7.2519 ,-0.6494)..
            ( 7.2254 ,-0.7071)..( 7.1949 ,-0.7604)..( 7.1605 ,-0.8090)..
            ( 7.1227 ,-0.8526)..( 7.0818 ,-0.8910)..( 7.0382 ,-0.9239)..
            ( 6.9925 ,-0.9511)..( 6.9448 ,-0.9724)..( 6.8959 ,-0.9877)..
            ( 6.8460 ,-0.9969)..( 6.7956 ,-1.0000);

   ph_prim := makepath makepen
            ( 5.5300 , 0.9724)..( 5.6275 , 0.9877)..( 5.7259 , 0.9969)..
            ( 5.8248 , 1.0000)..( 5.9237 , 0.9969)..( 6.0222 , 0.9877)..
            ( 6.1197 , 0.9724)..( 6.2158 , 0.9511)..( 6.3101 , 0.9239)..
            ( 6.4022 , 0.8910)..( 6.4916 , 0.8526)..( 6.5780 , 0.8090)..
            ( 6.6609 , 0.7604)..( 6.7400 , 0.7071)..( 6.8150 , 0.6494)..
            ( 6.8855 , 0.5878)..( 6.9513 , 0.5225)..( 7.0120 , 0.4540)..
            ( 7.0675 , 0.3827)..( 7.1177 , 0.3090)..( 7.1622 , 0.2334)..
            ( 7.2010 , 0.1564)..( 7.2341 , 0.0785)..( 7.2613 ,-0.0000)..
            ( 7.2826 ,-0.0785)..( 7.2981 ,-0.1564)..( 7.3078 ,-0.2334)..
            ( 7.3118 ,-0.3090)..( 7.3102 ,-0.3827)..( 7.3033 ,-0.4540)..
            ( 7.2910 ,-0.5225)..( 7.2738 ,-0.5878)..( 7.2519 ,-0.6494)..
            ( 7.2254 ,-0.7071)..( 7.1949 ,-0.7604)..( 7.1605 ,-0.8090)..
            ( 7.1227 ,-0.8526)..( 7.0818 ,-0.8910)..( 7.0382 ,-0.9239)..
            ( 6.9925 ,-0.9511)..( 6.9448 ,-0.9724)..( 6.8959 ,-0.9877)..
            ( 6.8460 ,-0.9969)..( 6.7956 ,-1.0000)..( 6.7453 ,-0.9969)..
            ( 6.6954 ,-0.9877)..( 6.6464 ,-0.9724);

   pi := makepath makepen
            ( 6.7956 ,-1.0000)..( 6.7453 ,-0.9969)..( 6.6954 ,-0.9877)..
            ( 6.6464 ,-0.9724)..( 6.5988 ,-0.9511)..( 6.5530 ,-0.9239)..
            ( 6.5095 ,-0.8910)..( 6.4686 ,-0.8526)..( 6.4308 ,-0.8090)..
            ( 6.3964 ,-0.7604)..( 6.3658 ,-0.7071)..( 6.3394 ,-0.6494)..
            ( 6.3175 ,-0.5878)..( 6.3002 ,-0.5225)..( 6.2880 ,-0.4540)..
            ( 6.2810 ,-0.3827)..( 6.2795 ,-0.3090)..( 6.2835 ,-0.2334)..
            ( 6.2932 ,-0.1564)..( 6.3086 ,-0.0785)..( 6.3300 ,-0.0000)..
            ( 6.3572 , 0.0785)..( 6.3902 , 0.1564)..( 6.4291 , 0.2334)..
            ( 6.4736 , 0.3090)..( 6.5237 , 0.3827)..( 6.5793 , 0.4540)..
            ( 6.6400 , 0.5225)..( 6.7058 , 0.5878)..( 6.7763 , 0.6494)..
            ( 6.8512 , 0.7071)..( 6.9304 , 0.7604)..( 7.0133 , 0.8090)..
            ( 7.0996 , 0.8526)..( 7.1891 , 0.8910)..( 7.2811 , 0.9239)..
            ( 7.3755 , 0.9511)..( 7.4716 , 0.9724)..( 7.5691 , 0.9877)..
            ( 7.6676 , 0.9969)..( 7.7664 , 1.0000);

   pi_prim := makepath makepen
            ( 6.9448 ,-0.9724)..( 6.8959 ,-0.9877)..( 6.8460 ,-0.9969)..
            ( 6.7956 ,-1.0000)..( 6.7453 ,-0.9969)..( 6.6954 ,-0.9877)..
            ( 6.6464 ,-0.9724)..( 6.5988 ,-0.9511)..( 6.5530 ,-0.9239)..
            ( 6.5095 ,-0.8910)..( 6.4686 ,-0.8526)..( 6.4308 ,-0.8090)..
            ( 6.3964 ,-0.7604)..( 6.3658 ,-0.7071)..( 6.3394 ,-0.6494)..
            ( 6.3175 ,-0.5878)..( 6.3002 ,-0.5225)..( 6.2880 ,-0.4540)..
            ( 6.2810 ,-0.3827)..( 6.2795 ,-0.3090)..( 6.2835 ,-0.2334)..
            ( 6.2932 ,-0.1564)..( 6.3086 ,-0.0785)..( 6.3300 ,-0.0000)..
            ( 6.3572 , 0.0785)..( 6.3902 , 0.1564)..( 6.4291 , 0.2334)..
            ( 6.4736 , 0.3090)..( 6.5237 , 0.3827)..( 6.5793 , 0.4540)..
            ( 6.6400 , 0.5225)..( 6.7058 , 0.5878)..( 6.7763 , 0.6494)..
            ( 6.8512 , 0.7071)..( 6.9304 , 0.7604)..( 7.0133 , 0.8090)..
            ( 7.0996 , 0.8526)..( 7.1891 , 0.8910)..( 7.2811 , 0.9239)..
            ( 7.3755 , 0.9511)..( 7.4716 , 0.9724)..( 7.5691 , 0.9877)..
            ( 7.6676 , 0.9969)..( 7.7664 , 1.0000)..( 7.8653 , 0.9969)..
            ( 7.9638 , 0.9877)..( 8.0613 , 0.9724);

   pj := makepath makepen
            ( 7.7664 , 1.0000)..( 7.8653 , 0.9969)..( 7.9638 , 0.9877)..
            ( 8.0613 , 0.9724)..( 8.1574 , 0.9511)..( 8.2517 , 0.9239)..
            ( 8.3438 , 0.8910)..( 8.4333 , 0.8526)..( 8.5196 , 0.8090)..
            ( 8.6025 , 0.7604)..( 8.6816 , 0.7071)..( 8.7566 , 0.6494)..
            ( 8.8271 , 0.5878)..( 8.8929 , 0.5225)..( 8.9536 , 0.4540)..
            ( 9.0092 , 0.3827)..( 9.0593 , 0.3090)..( 9.1038 , 0.2334)..
            ( 9.1427 , 0.1564)..( 9.1757 , 0.0785)..( 9.2029 ,-0.0000)..
            ( 9.2242 ,-0.0785)..( 9.2397 ,-0.1564)..( 9.2494 ,-0.2334)..
            ( 9.2534 ,-0.3090)..( 9.2519 ,-0.3827)..( 9.2449 ,-0.4540)..
            ( 9.2326 ,-0.5225)..( 9.2154 ,-0.5878)..( 9.1935 ,-0.6494)..
            ( 9.1670 ,-0.7071)..( 9.1365 ,-0.7604)..( 9.1021 ,-0.8090);

%%--------------------------------------------------------------------------
%%
%% ,.... the forward propagating optical fields inside the cavity....
%%
   oldahlength := ahlength;  %% ArrowHead length
   oldahangle  := ahangle;   %% ArrowHead angle
   u := 3mm;

% %%
% %% (Draw hidden parts of forward propagating left-handed helix / LCP)
% %%
%    Ta := identity shifted (14mm,25mm);
%    Th := identity scaled u xscaled 0.5 transformed Ta;
%    label.rt(btex
%             $E^{f+}_{\omega_1,\omega_2,\omega_3}$
%             etex, ((14mm,3mm)) transformed Ta);
%    pickup pencircle scaled ellipsethickness;
%    draw pb transformed Th withcolor black;
%    draw pd transformed Th withcolor black;
%    draw pf transformed Th withcolor black;
%    draw ph transformed Th withcolor black;
%    ahlength := 3pt;
%    %% drawarrow pj transformed Th withcolor black;
%    draw pj transformed Th withcolor black;
%    ahlength := oldahlength;
%    ahangle  := oldahangle;
% 
% %%
% %% (Draw k-vector / direction of propagation of forward propagating
% %%  left-handed helix / LCP)
% %%
%    linecap := butt;
%    pickup pencircle scaled whitegap;
%    drawarrow ((-5mm,0mm)--(18mm,0mm)) transformed Ta withcolor white;
%    pickup pencircle scaled arrowthickness;
%    drawarrow ((-5mm,0mm)--(18mm,0mm)) transformed Ta withcolor black;
% 
% %%
% %% (Draw in-front parts of forward propagating left-handed helix / LCP)
% %%
%    linecap := butt;
%    pickup pencircle scaled whitegap;
%    draw pa transformed Th withcolor white;
%    pickup pencircle scaled ellipsethickness;
%    draw pa_prim transformed Th withcolor black;
% 
%    linecap := butt;
%    pickup pencircle scaled whitegap;
%    draw pc transformed Th withcolor white;
%    pickup pencircle scaled ellipsethickness;
%    draw pc_prim transformed Th withcolor black;
% 
%    linecap := butt;
%    pickup pencircle scaled whitegap;
%    draw pe transformed Th withcolor white;
%    pickup pencircle scaled ellipsethickness;
%    draw pe_prim transformed Th withcolor black;
% 
%    linecap := butt;
%    pickup pencircle scaled whitegap;
%    draw pg transformed Th withcolor white;
%    pickup pencircle scaled ellipsethickness;
%    draw pg_prim transformed Th withcolor black;
% 
%    linecap := butt;
%    pickup pencircle scaled whitegap;
%    draw pi transformed Th withcolor white;
%    pickup pencircle scaled ellipsethickness;
%    draw pi_prim transformed Th withcolor black;
% 
% 
% %%
% %% (Draw hidden parts of forward propagating right-handed helix / RCP)
% %%
%    Ta := identity shifted (14mm,35mm);
%    Th := identity scaled u xscaled 0.5 yscaled -1.0 transformed Ta;
%    label.rt(btex
%             $E^{f-}_{\omega_1,\omega_2,\omega_3}$
%             etex, ((14mm,3mm)) transformed Ta);
%    pickup pencircle scaled ellipsethickness;
%    draw pb transformed Th withcolor black;
%    draw pd transformed Th withcolor black;
%    draw pf transformed Th withcolor black;
%    draw ph transformed Th withcolor black;
%    ahlength := 3pt;
%    %% drawarrow pj transformed Th withcolor black;
%    draw pj transformed Th withcolor black;
%    ahlength := oldahlength;
%    ahangle  := oldahangle;
% 
% %%
% %% (Draw k-vector / direction of propagation of forward propagating
% %%  right-handed helix / RCP)
% %%
%    linecap := butt;
%    pickup pencircle scaled whitegap;
%    drawarrow ((-5mm,0mm)--(18mm,0mm)) transformed Ta withcolor white;
%    pickup pencircle scaled arrowthickness;
%    drawarrow ((-5mm,0mm)--(18mm,0mm)) transformed Ta withcolor black;
% 
% %%
% %% (Draw in-front parts of forward propagating right-handed helix / RCP)
% %%
%    linecap := butt;
%    pickup pencircle scaled whitegap;
%    draw pa transformed Th withcolor white;
%    pickup pencircle scaled ellipsethickness;
%    draw pa_prim transformed Th withcolor black;
% 
%    linecap := butt;
%    pickup pencircle scaled whitegap;
%    draw pc transformed Th withcolor white;
%    pickup pencircle scaled ellipsethickness;
%    draw pc_prim transformed Th withcolor black;
% 
%    linecap := butt;
%    pickup pencircle scaled whitegap;
%    draw pe transformed Th withcolor white;
%    pickup pencircle scaled ellipsethickness;
%    draw pe_prim transformed Th withcolor black;
% 
%    linecap := butt;
%    pickup pencircle scaled whitegap;
%    draw pg transformed Th withcolor white;
%    pickup pencircle scaled ellipsethickness;
%    draw pg_prim transformed Th withcolor black;
% 
%    linecap := butt;
%    pickup pencircle scaled whitegap;
%    draw pi transformed Th withcolor white;
%    pickup pencircle scaled ellipsethickness;
%    draw pi_prim transformed Th withcolor black;

%%--------------------------------------------------------------------------
%%
%% .... the backward propagating optical fields outside the cavity ....
%%

%%
%% (Draw hidden parts of backward propagating left-handed helix / LCP)
%%
   Ta := identity shifted (-18mm,2mm);
   Th := identity scaled u xscaled 0.5 transformed Ta;
   label.lft(btex $A^+_{\rm c}(0)$ etex, ((0mm,1mm)) transformed Ta);
   pickup pencircle scaled ellipsethickness;
%   draw pb transformed Th withcolor black;
%   draw pd transformed Th withcolor black;
   draw pf transformed Th withcolor black;
   draw ph transformed Th withcolor black;
   draw pj transformed Th withcolor black;

%%
%% (Draw k-vector / direction of propagation of backward propagating
%%  left-handed helix / LCP)
%%
   linecap := butt;
   pickup pencircle scaled whitegap;
   drawarrow reverse ((0mm,0mm)--(16mm,0mm)) transformed Ta withcolor white;
   pickup pencircle scaled arrowthickness;
   drawarrow reverse ((0mm,0mm)--(16mm,0mm)) transformed Ta withcolor black;

%%
%% (Draw in-front parts of backward propagating left-handed helix / LCP)
%%
   linecap := butt;
   pickup pencircle scaled whitegap;
%   draw pa transformed Th withcolor white;
%   pickup pencircle scaled ellipsethickness;
%   ahlength := 3pt;
%   %% drawarrow reverse pa_prim transformed Th withcolor black;
%   draw pa_prim transformed Th withcolor black;
%   ahlength := oldahlength;
%   ahangle  := oldahangle;

%   linecap := butt;
%   pickup pencircle scaled whitegap;
%   draw pc transformed Th withcolor white;
%   pickup pencircle scaled ellipsethickness;
%   draw pc_prim transformed Th withcolor black;

   linecap := butt;
   pickup pencircle scaled whitegap;
   draw pe transformed Th withcolor white;
   pickup pencircle scaled ellipsethickness;
   draw pe_prim transformed Th withcolor black;

   linecap := butt;
   pickup pencircle scaled whitegap;
   draw pg transformed Th withcolor white;
   pickup pencircle scaled ellipsethickness;
   draw pg_prim transformed Th withcolor black;

   linecap := butt;
   pickup pencircle scaled whitegap;
   draw pi transformed Th withcolor white;
   pickup pencircle scaled ellipsethickness;
   draw pi_prim transformed Th withcolor black;

%%
%% (Draw hidden parts of backward propagating right-handed helix / RCP)
%%
   Ta := identity shifted (-18mm,2mm+7mm);
   Th := identity scaled u xscaled 0.5 yscaled -1.0 transformed Ta;
   label.lft(btex $A^-_{\rm c}(0)$ etex, ((0mm,1mm)) transformed Ta);
   pickup pencircle scaled ellipsethickness;
%   draw pb transformed Th withcolor black;
%   draw pd transformed Th withcolor black;
   draw pf transformed Th withcolor black;
   draw ph transformed Th withcolor black;
   draw pj transformed Th withcolor black;

%%
%% (Draw k-vector / direction of propagation of backward propagating
%%  right-handed helix / RCP)
%%
   linecap := butt;
   pickup pencircle scaled whitegap;
   drawarrow reverse ((0mm,0mm)--(16mm,0mm)) transformed Ta withcolor white;
   pickup pencircle scaled arrowthickness;
   drawarrow reverse ((0mm,0mm)--(16mm,0mm)) transformed Ta withcolor black;

%%
%% (Draw in-front parts of backward propagating right-handed helix / RCP)
%%
   linecap := butt;
   pickup pencircle scaled whitegap;
%   draw pa transformed Th withcolor white;
%   pickup pencircle scaled ellipsethickness;
%   ahlength := 3pt;
%   %% drawarrow reverse pa_prim transformed Th withcolor black;
%   draw pa_prim transformed Th withcolor black;
%   ahlength := oldahlength;
%   ahangle  := oldahangle;

%   linecap := butt;
%   pickup pencircle scaled whitegap;
%   draw pc transformed Th withcolor white;
%   pickup pencircle scaled ellipsethickness;
%   draw pc_prim transformed Th withcolor black;

   linecap := butt;
   pickup pencircle scaled whitegap;
   draw pe transformed Th withcolor white;
   pickup pencircle scaled ellipsethickness;
   draw pe_prim transformed Th withcolor black;

   linecap := butt;
   pickup pencircle scaled whitegap;
   draw pg transformed Th withcolor white;
   pickup pencircle scaled ellipsethickness;
   draw pg_prim transformed Th withcolor black;

   linecap := butt;
   pickup pencircle scaled whitegap;
   draw pi transformed Th withcolor white;
   pickup pencircle scaled ellipsethickness;
   draw pi_prim transformed Th withcolor black;

%%--------------------------------------------------------------------------
%%
%% .... the incident optical fields outside the cavity....
%%
   oldahlength := ahlength;  %% ArrowHead length
   oldahangle  := ahangle;   %% ArrowHead angle
   u := 3mm;

%%
%% (Draw hidden parts of left-handed helix outside the cavity.... / LCP)
%%
   Ta := identity shifted (-19mm,22mm);
   Th := identity scaled u xscaled 0.5 transformed Ta;
   label.bot(btex $A^+_{\rm p}(0)$, $A^+_{\rm s}(0)$ etex, ((2mm,-2.5mm)) transformed Ta);
   
   pickup pencircle scaled ellipsethickness;
%   draw pb transformed Th withcolor black;
%   draw pd transformed Th withcolor black;
   draw pf transformed Th withcolor black;
   draw ph transformed Th withcolor black;
   ahlength := 3pt;
   %% drawarrow pj transformed Th withcolor black;
   draw pj transformed Th withcolor black;
   ahlength := oldahlength;
   ahangle  := oldahangle;

%%
%% (Draw k-vector / direction of propagation of
%%  left-handed helix outside the cavity.... / LCP)
%%
   linecap := butt;
   pickup pencircle scaled whitegap;
   drawarrow ((1mm,0mm)--(17mm,0mm)) transformed Ta withcolor white;
   pickup pencircle scaled arrowthickness;
   drawarrow ((1mm,0mm)--(17mm,0mm)) transformed Ta withcolor black;

%%
%% (Draw in-front parts of left-handed helix outside the cavity.... / LCP)
%%
%   linecap := butt;
%   pickup pencircle scaled whitegap;
%   draw pa transformed Th withcolor white;
%   pickup pencircle scaled ellipsethickness;
%   draw pa_prim transformed Th withcolor black;

%   linecap := butt;
%   pickup pencircle scaled whitegap;
%   draw pc transformed Th withcolor white;
%   pickup pencircle scaled ellipsethickness;
%   draw pc_prim transformed Th withcolor black;

   linecap := butt;
   pickup pencircle scaled whitegap;
   draw pe transformed Th withcolor white;
   pickup pencircle scaled ellipsethickness;
   draw pe_prim transformed Th withcolor black;

   linecap := butt;
   pickup pencircle scaled whitegap;
   draw pg transformed Th withcolor white;
   pickup pencircle scaled ellipsethickness;
   draw pg_prim transformed Th withcolor black;

   linecap := butt;
   pickup pencircle scaled whitegap;
   draw pi transformed Th withcolor white;
   pickup pencircle scaled ellipsethickness;
   draw pi_prim transformed Th withcolor black;

%%
%% (Draw hidden parts of right-handed helix outside the cavity.... / RCP)
%%
   Ta := identity shifted (-19mm,22mm+7mm);
   Th := identity scaled u xscaled 0.5 yscaled -1.0 transformed Ta;
%   label.lft(btex $A^-_{\rm p}(0)$ etex, ((1mm,0mm)) transformed Ta);
   label.top(btex $A^-_{\rm p}(0)$, $A^-_{\rm s}(0)$ etex,
       ((2mm,2mm)) transformed Ta);

   pickup pencircle scaled ellipsethickness;
%   draw pb transformed Th withcolor black;
%   draw pd transformed Th withcolor black;
   draw pf transformed Th withcolor black;
   draw ph transformed Th withcolor black;
   ahlength := 3pt;
   %% drawarrow pj transformed Th withcolor black;
   draw pj transformed Th withcolor black;
   ahlength := oldahlength;
   ahangle  := oldahangle;

%%
%% (Draw k-vector / direction of propagation of
%%  right-handed helix outside the cavity.... / RCP)
%%
   linecap := butt;
   pickup pencircle scaled whitegap;
   drawarrow ((1mm,0mm)--(17mm,0mm)) transformed Ta withcolor white;
   pickup pencircle scaled arrowthickness;
   drawarrow ((1mm,0mm)--(17mm,0mm)) transformed Ta withcolor black;

%%
%% (Draw in-front parts of right-handed helix outside the cavity.... / RCP)
%%
%   linecap := butt;
%   pickup pencircle scaled whitegap;
%   draw pa transformed Th withcolor white;
%   pickup pencircle scaled ellipsethickness;
%   draw pa_prim transformed Th withcolor black;

%   linecap := butt;
%   pickup pencircle scaled whitegap;
%   draw pc transformed Th withcolor white;
%   pickup pencircle scaled ellipsethickness;
%   draw pc_prim transformed Th withcolor black;

   linecap := butt;
   pickup pencircle scaled whitegap;
   draw pe transformed Th withcolor white;
   pickup pencircle scaled ellipsethickness;
   draw pe_prim transformed Th withcolor black;

   linecap := butt;
   pickup pencircle scaled whitegap;
   draw pg transformed Th withcolor white;
   pickup pencircle scaled ellipsethickness;
   draw pg_prim transformed Th withcolor black;

   linecap := butt;
   pickup pencircle scaled whitegap;
   draw pi transformed Th withcolor white;
   pickup pencircle scaled ellipsethickness;
   draw pi_prim transformed Th withcolor black;


%%--------------------------------------------------------------------------
%%
%% .... the transmitted optical fields outside the cavity....
%%
   oldahlength := ahlength;  %% ArrowHead length
   oldahangle  := ahangle;   %% ArrowHead angle
   u := 3mm;

%%
%% (Draw hidden parts of left-handed helix outside the cavity.... / LCP)
%%
   Ta := identity shifted (51mm,12mm);
   Th := identity scaled u xscaled 0.5 transformed Ta;
   label.top(btex $A^+_{\rm p}(L)$, $A^+_{\rm s}(L)$ etex, ((11mm,-10mm))
       transformed Ta);
   pickup pencircle scaled ellipsethickness;
%   draw pb transformed Th withcolor black;
%   draw pd transformed Th withcolor black;
   draw pf transformed Th withcolor black;
   draw ph transformed Th withcolor black;
   ahlength := 3pt;
   %% drawarrow pj transformed Th withcolor black;
   draw pj transformed Th withcolor black;
   ahlength := oldahlength;
   ahangle  := oldahangle;

%%
%% (Draw k-vector / direction of propagation of
%%  left-handed helix outside the cavity.... / LCP)
%%
   linecap := butt;
   pickup pencircle scaled whitegap;
   drawarrow ((1mm,0mm)--(17mm,0mm)) transformed Ta withcolor white;
   pickup pencircle scaled arrowthickness;
   drawarrow ((1mm,0mm)--(17mm,0mm)) transformed Ta withcolor black;

%%
%% (Draw in-front parts of left-handed helix outside the cavity.... / LCP)
%%
%   linecap := butt;
%   pickup pencircle scaled whitegap;
%   draw pa transformed Th withcolor white;
%   pickup pencircle scaled ellipsethickness;
%   draw pa_prim transformed Th withcolor black;

%   linecap := butt;
%   pickup pencircle scaled whitegap;
%   draw pc transformed Th withcolor white;
%   pickup pencircle scaled ellipsethickness;
%   draw pc_prim transformed Th withcolor black;

   linecap := butt;
   pickup pencircle scaled whitegap;
   draw pe transformed Th withcolor white;
   pickup pencircle scaled ellipsethickness;
   draw pe_prim transformed Th withcolor black;

   linecap := butt;
   pickup pencircle scaled whitegap;
   draw pg transformed Th withcolor white;
   pickup pencircle scaled ellipsethickness;
   draw pg_prim transformed Th withcolor black;

   linecap := butt;
   pickup pencircle scaled whitegap;
   draw pi transformed Th withcolor white;
   pickup pencircle scaled ellipsethickness;
   draw pi_prim transformed Th withcolor black;

%%
%% (Draw hidden parts of right-handed helix outside the cavity.... / RCP)
%%
   Ta := identity shifted (51mm,12mm+7mm);
   Th := identity scaled u xscaled 0.5 yscaled -1.0 transformed Ta;
%   label.bot(btex $E^{T-}_{\omega_1,\omega_2,\omega_3}$
%             etex, ((6mm,9mm)) transformed Ta);
   label.bot(btex $A^-_{\rm p}(L)$, $A^-_{\rm s}(L)$ etex, ((11mm,9.5mm))
       transformed Ta);
   pickup pencircle scaled ellipsethickness;
%   draw pb transformed Th withcolor black;
%   draw pd transformed Th withcolor black;
   draw pf transformed Th withcolor black;
   draw ph transformed Th withcolor black;
   ahlength := 3pt;
   %% drawarrow pj transformed Th withcolor black;
   draw pj transformed Th withcolor black;
   ahlength := oldahlength;
   ahangle  := oldahangle;

%%
%% (Draw k-vector / direction of propagation of
%%  right-handed helix outside the cavity.... / RCP)
%%
   linecap := butt;
   pickup pencircle scaled whitegap;
   drawarrow ((1mm,0mm)--(17mm,0mm)) transformed Ta withcolor white;
   pickup pencircle scaled arrowthickness;
   drawarrow ((1mm,0mm)--(17mm,0mm)) transformed Ta withcolor black;

%%
%% (Draw in-front parts of right-handed helix outside the cavity.... / RCP)
%%
%   linecap := butt;
%   pickup pencircle scaled whitegap;
%   draw pa transformed Th withcolor white;
%   pickup pencircle scaled ellipsethickness;
%   draw pa_prim transformed Th withcolor black;

%   linecap := butt;
%   pickup pencircle scaled whitegap;
%   draw pc transformed Th withcolor white;
%   pickup pencircle scaled ellipsethickness;
%   draw pc_prim transformed Th withcolor black;

   linecap := butt;
   pickup pencircle scaled whitegap;
   draw pe transformed Th withcolor white;
   pickup pencircle scaled ellipsethickness;
   draw pe_prim transformed Th withcolor black;

   linecap := butt;
   pickup pencircle scaled whitegap;
   draw pg transformed Th withcolor white;
   pickup pencircle scaled ellipsethickness;
   draw pg_prim transformed Th withcolor black;

   linecap := butt;
   pickup pencircle scaled whitegap;
   draw pi transformed Th withcolor white;
   pickup pencircle scaled ellipsethickness;
   draw pi_prim transformed Th withcolor black;

   label(btex $\chi^{(2)}$ etex,(25mm,24mm));

%%
%% Draw the static magnetic field $H_0$ with labels...
%%
%   pickup pencircle scaled 0.4pt;
%   lineblackness:=1.0;
%   for lineheight = 0mm step 10mm until 40mm:
%      drawarrow ((5mm,lineheight)--(15mm,lineheight))
%            dashed evenly withcolor lineblackness [white,black];
%      drawarrow ((15mm,lineheight)--(35mm,lineheight))
%            dashed evenly withcolor lineblackness [white,black];
%      draw ((35mm,lineheight)--(45mm,lineheight))
%            dashed evenly withcolor lineblackness [white,black];
%   endfor
%   label.top(btex ${\bf H}_0$ etex, (10mm,40mm));

%%
%% Label the mirrors...
%%
%   label.top(btex $M_0$ etex, (-4mm,38mm));
%   label.top(btex $M_1$ etex, (54mm,38mm));
%   label.top(btex $\rho^{(0)}_{\pm}$ etex, (10mm,-6mm));
%   label.top(btex $\rho^{(1)}_{\pm}$ etex, (40mm,-6mm));


%   draw ((0mm,0mm)--(102mm,0mm)--(102mm,46mm)--(0mm,46mm)--(0mm,0mm)) shifted (-28mm,-9mm);
   endfig;
end

