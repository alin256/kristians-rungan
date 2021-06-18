rm *_r.png
finish=120
for i in {1..20};
do
convert resisitivity_anim_$[finish-i].png  -resize 1280x720  resisitivity_anim_$[finish+i]_r.png
done
for i in {100..120};
do
convert resisitivity_anim_${i}.png  -resize 1280x720  resisitivity_anim_${i}_r.png
done
convert -delay 20 -loop 0 *_r.png prior.gif