use std::time::Instant;

use image::{imageops::FilterType, Rgb, RgbImage};
use ndarray::Array4;
use ort::{
    session::{builder::GraphOptimizationLevel, Session},
    value::TensorRef,
};

const H: usize = 128;
const W: usize = 512;

fn ms(t: Instant) -> f64 {
    t.elapsed().as_secs_f64() * 1000.0
}

fn preprocess(path: &str) -> Result<Array4<f32>, Box<dyn std::error::Error>> {
    let img = image::open(path)?.to_rgb8();
    let resized = image::imageops::resize(&img, W as u32, H as u32, FilterType::CatmullRom);
    Ok(Array4::<f32>::from_shape_fn((1, H, W, 3), |(_, y, x, c)| {
        resized.get_pixel(x as u32, y as u32)[c] as f32 / 127.5 - 1.0
    }))
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let model_path = args.get(1).map(String::as_str).unwrap_or("gen.onnx");
    let image_path = args.get(2).map(String::as_str).unwrap_or("test.png");
    let t_load = Instant::now();
    let environment = ort::init().build()?;
    let mut session = Session::builder(&environment)?
        .with_optimization_level(GraphOptimizationLevel::Level3)?
        .with_intra_threads(4)?
        .commit_from_file(model_path)?;
    println!("load model: {:.0} ms", ms(t_load));


    let input = preprocess(image_path)?;
    for _ in 0..5 {
        session.run(ort::inputs![TensorRef::from_array_view(&input)?])?;
    }
    let mut ts = Vec::new();
    for _ in 0..30 {
        let t = Instant::now();
        let outputs = session.run(ort::inputs![TensorRef::from_array_view(&input)?])?;
        let _ = outputs[0].try_extract_array::<f32>()?;
        ts.push(ms(t));
    }
    ts.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = ts.iter().sum::<f64>() / ts.len() as f64;
    println!(
        "inferensi (30x): {:.0} ms",
        mean
    );

    let t_total = Instant::now();

    let t = Instant::now();
    let input = preprocess(image_path)?;
    let t_pre = ms(t);

    let t = Instant::now();
    let outputs = session.run(ort::inputs![TensorRef::from_array_view(&input)?])?;
    let y = outputs[0].try_extract_array::<f32>()?; // [1, 128, 512, 3]
    let t_inf = ms(t);

    let t = Instant::now();
    let mut out = RgbImage::new(W as u32, H as u32);
    for yy in 0..H {
        for xx in 0..W {
            let px = [0usize, 1, 2]
                .map(|c| ((y[[0, yy, xx, c]] + 1.0) * 127.5).clamp(0.0, 255.0) as u8);
            out.put_pixel(xx as u32, yy as u32, Rgb(px));
        }
    }
    let t_post = ms(t);

    let t = Instant::now();
    let luma = image::imageops::grayscale(&out).into_raw();
    let decoded = rxing::helpers::detect_in_luma(luma, W as u32, H as u32, None);
    let t_dec = ms(t);

    let total = ms(t_total);

    out.save("restored.png")?;
    match decoded {
        Ok(r) => println!("\nDecode: {} (format: {:?})", r.getText(), r.getBarcodeFormat()),
        Err(e) => println!("\nFailed Decode: {:?}", e),
    }
    println!("\npreprocess  : {:>7.1} ms", t_pre);
    println!("inferensi   : {:>7.1} ms", t_inf);
    println!("postprocess : {:>7.1} ms", t_post);
    println!("decode rxing: {:>7.1} ms", t_dec);
    println!("TOTAL       : {:>7.1} ms", total);
    Ok(())
}