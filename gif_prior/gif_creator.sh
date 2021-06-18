rm *_r.png
finish=139
for i in {1..39};
do
cp resisitivity_anim_$[finish-i].png  resisitivity_anim_$[finish+i].png
echo 'copying ' $[finish-i] ' to ' $[finish+i]
done
convert -delay 20 -loop 0 *.png prior.gif