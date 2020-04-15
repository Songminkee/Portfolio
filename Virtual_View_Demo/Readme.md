![](./gif/Demo.gif)

- You can control using keys (w,s,a,d or arrow keys) in opencv window
- And You want quit, then please press 'ESC' in opencv window



### Environment

---

- tensorflow gpu >= 1.10
- opencv-python



### Data

---

- [Pretrained Weight](https://drive.google.com/file/d/1lYSNQbaQxIa5KKUCtyFhZqlz8v1dy54F/view)
- [Input example](https://drive.google.com/open?id=1M-5Bvwvw9Hl5DOwRNfYQ3QD1RSM0Pbtw)



### Demo

---

```
python Demo.py --synthesis_height 768 --synthesis_width 1024 --demo_tum 7 --demo_path ./example/ --checkpoint_path ./model/weight
```

- `synthesis_height`  : Height of window
- `synthesis_width` : Width of window
- `demo_path` : Root path of Input images
- `checkpoint_path` : Path of weight
- `demo_tum` : Number of synthesis view to be created between input image



### Reference

---

- Unsupervised Monocular Depth Estimation with Left-Right Consistency_CVPR 2017 [[github]](https://github.com/mrharicot/monodepth)