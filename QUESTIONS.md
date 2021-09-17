# Questions and Curious that appear on looking at this work.

At first glance it appears that Dino finds the self information of the image; offering some surprising consistency in video segmentation--as shown on the paper's examples--even when each frame is individually processed.  

Questions on looking at this work:

1. What is it learning? -- is it akin to a laplacian pyramid across the dataset unbound by pixel coordinates?

2. Can it be leveraged on downstream tasks and how? -- would it be useful for coordinate regression?
    The teacher student training approach of this model opens many possibilities. <br>
    So as to be useful for specific tasks - e.g. segmenting humans -- the project would probably require further training of specific heads.
    In the current form it is biased towards both subjects in the form of humans and animals and manmade objects. On very busy images the current training needs to be disentangled. 

3. What are its failure points? Are there biases that condition results, is it succeptible to adversaries?
    This question is important to both design stronger self supervision and to understand if biases will creep on downstream tasks. 
    From a few tests it appears to be biased to sharp focus, sharper objects, outlier statistcal modes.

## Biases and Characteristics of pretrained Dino

<details>
  <summary>  Focus Bias appears strongly in Dino, this is probably inherent to the architecture. <br> While attention heads have less dependence on pixel neighborhood thatn convolutions, weigths still are triggered by distinct pixel steps <br>
    <img width="5%" src=".github/bangladesh_factory_attn.jpg"/> </summary>

  <div align="center">
<td> <img width="50%" src=".github/bangladesh_factory_attn.jpg"/> 
</td>
<a href='https://i.guim.co.uk/img/media/5ef7400158bf88db31347de8e6bb023d5a443f13/0_230_5649_3390/master/5649.jpg?width=1920&quality=85&auto=format&fit=max&s=1ab893b235820659589fa2c786e4d5f6'>Image source</a>
</div>
</details>

<details>
  <summary> Center Bias<br> was tested by running Dino, rolling the image and rerunning it. Not only the balance of attention changes but also its absolute values. This perhaps could be trained against.<br>
  <img width="5%" src=".github/lava_ori.jpg"/>
  <img width="5%" src=".github/lava_rolled.jpg"/> 
  <img width="5%" src=".github/urgencies_attn.jpg"/> 
  <img width="5%" src=".github/urgencies_roll_attn.jpg"/> 
  
  </summary>
<div align="center">
<table><tr>
<td> <img width="100%" src=".github/lava_ori.jpg"/> </td>
<td> <img width="100%" src=".github/lava_rolled.jpg"/></td>

<td> <img width="100%" src=".github/urgencies_attn.jpg"/> </td>
<td> <img width="100%" src=".github/urgencies_roll_attn.jpg"/>
</td>
</tr></table>
</div>
<div align="left">
<a href='https://www.theguardian.com/world/gallery/2020/jan/13/lava-gushes-from-taal-volcano-in-philippines-in-pictures#img-8'>Image1 source</a>
</div>
<div align="right">
<a href='https://www.theguardian.com/artanddesign/gallery/2020/apr/04/20-photographs-of-the-week#img-12'>Image2 source</a>

</div>

</details>

<details>
  <summary> Intrusion of Image Artifacts <br>
  Dino appears to excell at finding what does not belong in the image. When artifacts appear, they can become center of attention.<br>
  
  <img width="5%" src=".github/sunset.jpg"/> <img width="4.55%" src=".github/sunset_crop.jpg"/>
  <img width="4.55%" src=".github/prix-pictet-2019-shortlist-photo-essay_img-23_attn.jpg"/>
  </summary>

Image of sunset has artifacts from poor video compression, which dissappear when cropped.<br>
On the corrupted film, the human is not highlit, while artifacts are included the highest magnitude of attention picks up brick patterns.
<div align="center">
<table><tr>

<td> <img width="100%"  src=".github/sunset.jpg"/> </td>
<td> <img width="91%"  src=".github/sunset_crop.jpg"/> </td>
<br>
<td> <img width="100%"  src=".github/prix-pictet-2019-shortlist-photo-essay_img-23_attn.jpg"/>
<a href='https://www.theguardian.com/artanddesign/2019/jul/05/prix-pictet-2019-shortlist-photo-essay#img-23'>Image source</a>
</td>
</tr></table>
</div>
</details>

<details>
    <summary> Excells separating subjects from natural textures, even when those are sharp. Straight, even thickness lines are attract generally simlar magnitude of attention. <br>
     <img width="5%" src=".github/subjects_nature.jpg">
     <img width="5%" src=".github/death-toll-from-floods-in-south-asia-rises-to-more-than-100_img-2_attn.jpg">
     <img width="5%" src=".github/bourse_europeene.jpg">
     </summary>

<div align="center">
<table><tr>
<td> <img width="100%"  src=".github/subjects_nature.jpg"> 
<a href='https://www.theguardian.com/artanddesign/gallery/2020/jun/06/20-photographs-of-the-week#img-20'>Image source</a></td>
</td>
<td> <img width="100%"  src=".github/death-toll-from-floods-in-south-asia-rises-to-more-than-100_img-2_attn.jpg">
<a href='https://www.theguardian.com/environment/2019/jul/16/death-toll-from-floods-in-south-asia-rises-to-more-than-100#img-2'>Image source</a></td>
<td> <img width="100%"  src=".github/bourse_europeene.jpg"/>
<a href='https://www.lapresse.ca/affaires/marches/201907/04/01-5232681-les-bourses-europeennes-sans-direction-en-labsence-de-wall-street.php'>Image source</a></td>
</tr></table>
</div>

</details>



<details>
    <summary>Glasses, tight patterns and unnatural colors appear to attract most attention. <br>
     <img width="5%" src=".github/do-i-really-care-woody-allen-comes-out-fighting_img-1_attn.jpg">
     <img width="5%" src=".github/a-lost-elephant-and-hong-kong-protests-wednesdays-best-photos_img-10_attn.jpg">
     <img width="5%" src=".github/trump-roger-stone-sentencing-reaction-criticism_img-1_attn.jpg">
     </summary>
<div align="center">
<table><tr>
<td> <img width="100%"  src=".github/do-i-really-care-woody-allen-comes-out-fighting_img-1_attn.jpg">
<a href='https://www.theguardian.com/film/2020/may/29/do-i-really-care-woody-allen-comes-out-fighting#img-1'>Image source</a>
</td>
<td> <img width="100%"  src=".github/a-lost-elephant-and-hong-kong-protests-wednesdays-best-photos_img-10_attn.jpg"/> 
<a href='https://www.theguardian.com/media/gallery/2019/jun/12/a-lost-elephant-and-hong-kong-protests-wednesdays-best-photos#img-10'>Image source</a>
</td>
<td> <img width="100%"  src=".github/trump-roger-stone-sentencing-reaction-criticism_img-1_attn.jpg"/>
<a href="https://www.theguardian.com/us-news/2020/feb/11/trump-roger-stone-sentencing-reaction-criticism#img-1">Image source</a>
</td>
</tr></table>
</div>



</details>


## Code modifications are simple condensation of existing code
Images shown here were processed by
```python
from x_infer_simple import InferDino, show
D = InferDino() 
D.run(<imagename>, crop=(y0,y1,x0,x1))
show(D.lump( save=<filename>))
# or
D.batch(input_folder=<>, output_folder=<>)
```
or
```bash
# requires ffmpeg
python x_video_generation.py --input_path <str> --output_path <str> --frames <int_start int_end> --as_video 1 --crop <ints y0 y1 x0 x1>
```
