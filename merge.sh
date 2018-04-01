#!/usr/bin/env bash

head="image_id,image_category,neckline_left,neckline_right,center_front,shoulder_left,shoulder_right,armpit_left,armpit_right,waistline_left,waistline_right,cuff_left_in,cuff_left_out,cuff_right_in,cuff_right_out,top_hem_left,top_hem_right,waistband_left,waistband_right,hemline_left,hemline_right,crotch,bottom_left_in,bottom_left_out,bottom_right_in,bottom_right_out"

echo $head > res.csv
cat blouse.csv >> res.csv
cat skirt.csv >> res.csv
cat outwear.csv >> res.csv
cat dress.csv >> res.csv
cat trousers.csv >> res.csv
