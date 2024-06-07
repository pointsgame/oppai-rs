use anyhow::{anyhow, Result};
use oppai_field::extended_field::ExtendedField;
use oppai_svg::Config;
use resvg::tiny_skia::Pixmap;
use resvg::usvg::Tree;

pub fn render(field: &ExtendedField, config: &Config) -> Result<()> {
  let svg = oppai_svg::field_to_svg(config, field);
  let tree = Tree::from_str(&svg.to_string(), &Default::default())?;
  let mut pixmap = Pixmap::new(config.width, config.height).ok_or(anyhow!("Failed to create pixmap"))?;
  resvg::render(&tree, Default::default(), &mut pixmap.as_mut());
  let png = pixmap.encode_png()?;
  let image = image::load_from_memory(&png)?;
  viuer::print(
    &image,
    &viuer::Config {
      width: Some(config.width / 8),
      ..Default::default()
    },
  )?;
  Ok(())
}
