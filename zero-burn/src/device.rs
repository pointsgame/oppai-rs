use burn::tensor::{Device, DeviceType};
use thiserror::Error;

#[derive(Error, Debug)]
pub enum DeviceError {
  #[error("invalid device index '{0}'")]
  InvalidIndex(String),
  #[error("unknown device '{name}', available backends: auto, {available}")]
  UnknownBackend { name: String, available: String },
  #[error("device {name}:{index} not found ({count} available)")]
  NotFound { name: String, index: usize, count: usize },
}

/// Help text for a command line option parsed with [`parse_device`].
pub const DEVICE_HELP: &str = "Compute device: `auto` (first available GPU, then CPU) or a backend with an optional index into its devices, e.g. `cuda:1` or `vulkan:0`";

/// Backends enabled at build time, by the name used on the command line.
fn device_types() -> Vec<(&'static str, DeviceType)> {
  vec![
    #[cfg(feature = "cuda")]
    ("cuda", DeviceType::Cuda),
    #[cfg(feature = "rocm")]
    ("rocm", DeviceType::Rocm),
    #[cfg(feature = "vulkan")]
    ("vulkan", DeviceType::Vulkan),
    #[cfg(feature = "webgpu")]
    ("webgpu", DeviceType::WebGpu),
    #[cfg(feature = "flex")]
    ("flex", DeviceType::Flex),
  ]
}

/// The device a command line names: `auto` for the first available GPU, then
/// the CPU, or a backend with an optional index into its devices, e.g. `cuda:1`
/// or `vulkan:0` (see [`list_devices`]).
pub fn parse_device(device: &str) -> Result<Device, DeviceError> {
  if device == "auto" {
    return Ok(Device::default());
  }
  let (name, index) = match device.split_once(':') {
    Some((name, index)) => (
      name,
      index
        .parse::<usize>()
        .map_err(|_| DeviceError::InvalidIndex(index.to_string()))?,
    ),
    None => (device, 0),
  };
  let Some((_, device_type)) = device_types().into_iter().find(|(n, _)| *n == name) else {
    let names: Vec<_> = device_types().into_iter().map(|(n, _)| n).collect();
    return Err(DeviceError::UnknownBackend {
      name: name.to_string(),
      available: names.join(", "),
    });
  };
  let devices = Device::enumerate(device_type).into_vec();
  let count = devices.len();
  devices.into_iter().nth(index).ok_or_else(|| DeviceError::NotFound {
    name: name.to_string(),
    index,
    count,
  })
}

/// Print every device [`parse_device`] accepts, one per line.
pub fn list_devices() {
  for (name, device_type) in device_types() {
    for (index, device) in Device::enumerate(device_type).into_iter().enumerate() {
      println!("{name}:{index}\t{device:?}");
    }
  }
}
