rm *_r.png
finish=139
for i in {100..139};
do
convert resisitivity_anim_${i}.png  -resize 1280x720  resisitivity_anim_${i}_r.png
echo 'converting'${i}
done
for i in {1..39};
do
convert resisitivity_anim_$[finish-i].png  -resize 1280x720  resisitivity_anim_$[finish+i]_r.png
echo 'converting'$[finish+i]
done
convert -delay 20 -loop 0 *_r.png posterior.gif