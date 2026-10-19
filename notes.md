# How model work? 

### Big Picture 
Basically, this model is an image-to-image translator. It takes a damaged barcode image and outputs a cleaned-up barcode image of the same size. It doesn't read the barcode itself. Reading is done afterwards bby ZBar and ZXing, which are ordinary decoder libraries. So the pipeline has 2 separate parts: a neural network that repairs the picture, and a decoder that tries to read the repaired picture. The decoders are never part of the network, and no gradient flows through them. They only influence traiining indirectly, by deciding which training pairs get sampled more often and which checkpoint keep. 

### Image Input 
Whatever image i give the model, it first resized to 128 px high and 512 px wide, with 3 channels, and its pixel values are mapped from 0-255 into the range -1 to 1. That rage matches the final tanh layer of the generator, which can only output values between -1 and 1.


### The generator: U-Net 
The generator is a U-Net, which has an encoder that compresses the image, a decoder that expands it again, and skip connection between matching levels. The encoder is 7 blocks, each halving height and width while increasing the number of channels. The tensor goes from 128x512x3 to 64x256x64, then 32x128x128, 16x64x256, 8x32x512, 4x16x512, and finally a 1x4x512 bottleneck. By the bottleneck each of the 4 positions summarizes a very wide horizontal stretch of the original image. This matters for barcode because the meaning of a bar depends on its width relative to the other across the whole code. The network therefore has a way to reason globally, not just about local pixels. 

The decoder mirror this. Each block doubles the resolution with `UpSampling2D` and then applies a separable convolution, bringing the tensor back up through 2x8, 4x16, 8x32, 16x64, 32x128, and 64x256. A last block takes it to 128x512 with 32 channels, and a final 3x3 convolution with tanh produces the 3 channel output image. 

The skip connection are what make a U-Net suited to this task. After each decoder blok, the output it concatenated with the encoder feature map of the same resolution. The bottleneck is very compressed, so it can't carry the exact positions of thinbar edges on its own. The skip hand the decoder high-resolution detail from the encoder, so it can combine "what the whole barcode looks like" with "where exactly the edges are".

**Why separable convolutions?**
A normal conv mixes spatial and channel information in a single large set of weights. A separable conv splits this into a depthwise step, which filters each channel spatially on its own, and a pointwise step, which mixes channel with a 1x1 convolution. This gives similiar behaviour with far fewer weights, which is why the paper calls the model lightweight and why my version has 2.73M parameters. Between conv i use instance normalization, which normalizes each image on its own, so one image with unusual contrast can't distort other. The encoder uses LeakyReLU and the decoder uses ReLU. Dropout of 0.5 is applied in the first two decoder blocks, as in standart Pix2Pix, to stop the network from relying too heavilly on individual bottleneck features. 

### How it learns?
Since i turned the GAN off, training is just plain supervised regression. For each damaged input the network makes a restored image, and I punish it based on how far that image is from the clean target. The punishment has 3 parts,and each one covers a different  way of being wrong. 

The first is MAE, which is the average absolute pixel difference over the whole image. It keeps overall brightness and layout roughly right. The problem is that it's dominated by the big white background, so on almost all-white output would already look okay by this metric. 

The second is FMAE, which is the same error but only counted inside a mask around the bars. I build the mask from the clean target by marking dark pixels as bars and then dilating them with a 15x15 window, so it also covers a little margin of background next to each bar. It will forces the network to care about the bars, which are the only part that actually carries informations. 

The third is the SSIM term. It compares local structure (contrast and pattern) insted of raw pixel values, so it rewards getting the bar pattern right even if the intensities are slightly off. That's closer to what a decoder actually cares about. 

Without a discriminator, nothing is directly pushing the outputs to look sharp. A pure regression loss tends to average between several plausible answers, which gives soft, blurry edges. The decode aware part kind of makes up for that, because a blurry, ambiguous edge fails to decode and gets noticed. 


### How the decoders shape training
Before training, i decode every clean target once and store the text. Then every couple of epochs i run the current generator over the all the training inputs and decode what comes out. A pair counts as failed if neither ZBar nor ZXing gives back the same text as the clean target. Failed pairs get a sampling weight of 3, so in the following epochs the network sees its hard cases more often. After each epoch i also measure the decode rate on the val set, and the checkpoint i save as `gen_best_dec` is the one with the highest decode rate. So the network itself is still trained only through pixel loss, but the data it sees and the checkpoint i keep are both picked by what actually decodes. 

### Environment i used 
- Tensorflow 2.19.0, GPU Nvidia GeForce RTX 3090 (about 22.3 GB usable), cuDNN 9.3. 
- Libraries: numpy, pandas, matplotlib, pillow, `zxing-cpp`, `pyzbar`, opencv-python-headless 
- Seed 42 for everything. 

### Training Configuration 
- Adam, lr 2e-4, β1 = 0.5, batch size 1, 10 epochs
- Checkpoints: gen_last, gen_best (lowest val reconstruction loss), gen_best_dec (best decode rate, ties broken by val loss). Training can resume from state.json