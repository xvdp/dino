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
<td> <img width="50%" src=".github/bangladesh_factory_attn.jpg"/> </td>
</div>
</details>

<details>
  <summary> Center Bias<br> was tested by running Dino, rolling the image and rerunning it. Not only the balance of attention changes but also its absolute values. This perhaps could be trained against i<br>
  <img width="5%" src=".github/lava_ori.jpg"/>
  <img width="5%" src=".github/lava_rolled.jpg"/> 
  <img width="5%" src=".github/urgencies_attn.jpg"/> 
  <img width="5%" src=".github/urgencies_roll_attn.jpg"/> 
  
  </summary>
<div align="center">
<table><tr>
<td> <img width="100%" src=".github/lava_ori.jpg"/> </td>
<td> <img width="100%" src=".github/lava_rolled.jpg"/> </td>

<td> <img width="100%" src=".github/urgencies_attn.jpg"/> </td>
<td> <img width="100%" src=".github/urgencies_roll_attn.jpg"/> </td>
</tr></table>
</div>

[Image Source](https://www.theguardian.com/world/gallery/2020/jan/13/lava-gushes-from-taal-volcano-in-philippines-in-pictures#img-8)
</details>

<details>
  <summary> Intrusion of Image Artifacts <br>
  Dino appears to excell at finding what does not belong in the image. When artifacts appear, they can become center of attention.<br>
  <img width="5%" src=".github/sunset.jpg"/> <img width="4.55%" src=".github/sunset_crop.jpg"/>
  </summary>

<div align="center">
<table><tr>
<td> <img width="100%"  src=".github/sunset.jpg"/> </td>
<td> <img width="91%"  src=".github/sunset_crop.jpg"/> </td>
</tr></table>
</div>
</details>


<details>
    <summary> Excells separating subjects from natural textures. <br>
     <img width="5%" src=".github/subjects_nature.jpg">
     <img width="5%" src=".github/bourse_europeene.jpg">
     </summary>

<div align="center">
<table><tr>
<td> <img width="90%"  src=".github/subjects_nature.jpg"> </td>
<td> <img width="100%"  src=".github/bourse_europeene.jpg"/> </td>
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
