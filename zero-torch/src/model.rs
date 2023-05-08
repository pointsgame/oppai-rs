use std::borrow::Cow;
use std::marker::PhantomData;
use std::path::PathBuf;
use std::sync::Arc;

use indoc::indoc;
use ndarray::{Array1, Array3, Array4};
use num_traits::Float;
use numpy::array::{PyArray1, PyArray3};
use numpy::{Element, IntoPyArray};
use oppai_zero::model::{Model, TrainableModel};
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyDict};

const OPPAI_NET: &str = include_str!("../oppai_net.py");

pub trait DType {
  fn dtype() -> &'static str;
}

impl DType for f32 {
  fn dtype() -> &'static str {
    "torch.float32"
  }
}

impl DType for f64 {
  fn dtype() -> &'static str {
    "torch.float64"
  }
}

pub struct PyModel<N> {
  phantom: PhantomData<N>,
  path: Arc<PathBuf>,
  device: Arc<Cow<'static, str>>,
  model: PyObject,
  optimizer: PyObject,
}

impl<N: DType> PyModel<N> {
  pub fn new(path: PathBuf, width: u32, height: u32, channels: u32) -> PyResult<Self> {
    Python::with_gil(|py| {
      let oppai_net = PyModule::from_code(py, OPPAI_NET, "oppai_net.py", "oppai_net")?;
      let locals = [("torch", py.import("torch")?), ("oppai_net", oppai_net)].into_py_dict(py);
      locals.set_item("width", width)?;
      locals.set_item("height", height)?;
      locals.set_item("channels", channels)?;
      let dtype = py.eval(N::dtype(), None, Some(locals))?;
      locals.set_item("dtype", dtype)?;
      let model: PyObject = py
        .eval(
          "oppai_net.OppaiNet(width, height, channels).to(dtype)",
          None,
          Some(locals),
        )?
        .extract()?;
      locals.set_item("model", &model)?;
      let optimizer: PyObject = py
        .eval(
          "torch.optim.AdamW(model.parameters(), weight_decay = 1e-4)",
          None,
          Some(locals),
        )?
        .extract()?;

      Ok(Self {
        phantom: PhantomData::default(),
        path: Arc::new(path),
        device: Arc::new(Cow::Borrowed("cpu")),
        model,
        optimizer,
      })
    })
  }

  pub fn load(&self) -> PyResult<()> {
    PyModel::<N>::transfer(&self.model, "cpu")?;

    Python::with_gil(|py| {
      let locals = PyDict::new(py);
      locals.set_item("torch", py.import("torch")?)?;
      locals.set_item("model", &self.model)?;
      locals.set_item("optimizer", &self.optimizer)?;
      locals.set_item("path", self.path.as_ref())?;

      py.run(
        indoc! {"
          checkpoint = torch.load(path)
          model.load_state_dict(checkpoint['model_state_dict'])
          optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        "},
        None,
        Some(locals),
      )
    })?;

    PyModel::<N>::transfer(&self.model, self.device.as_ref())
  }

  pub fn try_clone(&self) -> PyResult<Self> {
    PyModel::<N>::transfer(&self.model, "cpu")?;

    let result = Python::with_gil(|py| -> PyResult<PyModel<N>> {
      let locals = PyDict::new(py);
      locals.set_item("copy", py.import("copy")?)?;
      locals.set_item("torch", py.import("torch")?)?;
      locals.set_item("model", &self.model)?;

      let model: PyObject = py.eval("copy.deepcopy(model)", None, Some(locals))?.extract()?;

      locals.set_item("model", &model)?;
      let optimizer: PyObject = py
        .eval("torch.optim.Adam(model.parameters())", None, Some(locals))?
        .extract()?;

      locals.set_item("old_optimizer", &self.optimizer)?;
      locals.set_item("new_optimizer", &optimizer)?;
      py.run(
        "new_optimizer.load_state_dict(old_optimizer.state_dict())",
        None,
        Some(locals),
      )?;

      Ok(Self {
        phantom: self.phantom,
        path: self.path.clone(),
        device: self.device.clone(),
        model,
        optimizer,
      })
    })?;

    PyModel::<N>::transfer(&self.model, self.device.as_ref())?;
    PyModel::<N>::transfer(&result.model, result.device.as_ref())?;

    Ok(result)
  }

  fn transfer(model: &PyObject, device: &str) -> PyResult<()> {
    Python::with_gil(|py| {
      let locals = PyDict::new(py);
      locals.set_item("model", model)?;
      locals.set_item("device", device)?;

      py.run("model.to(device)", None, Some(locals))
    })
  }

  pub fn to_device(&mut self, device: Cow<'static, str>) -> PyResult<()> {
    self.device = Arc::new(device);

    PyModel::<N>::transfer(&self.model, self.device.as_ref())
  }
}

impl<N: Float + Element + DType> Model<N> for PyModel<N> {
  type E = PyErr;

  fn predict(&self, inputs: Array4<N>) -> Result<(Array3<N>, Array1<N>), Self::E> {
    Python::with_gil(|py| {
      let locals = PyDict::new(py);
      locals.set_item("torch", py.import("torch")?)?;
      locals.set_item("inputs", inputs.into_pyarray(py))?;
      locals.set_item("model", &self.model)?;
      locals.set_item("device", self.device.as_ref())?;

      py.run("model.eval()", None, Some(locals))?;
      py.run(
        "policies, values = map(lambda x : x.detach().cpu().numpy(), model.predict(torch.from_numpy(inputs).to(device)))",
        None,
        Some(locals),
      )?;

      let policies: &PyArray3<N> = locals.get_item("policies").unwrap().extract()?;
      let values: &PyArray1<N> = locals.get_item("values").unwrap().extract()?;

      Ok((
        policies.readonly().as_array().to_owned(),
        values.readonly().as_array().to_owned(),
      ))
    })
  }
}

impl<N: Float + Element + DType> TrainableModel<N> for PyModel<N> {
  type TE = Self::E;

  fn train(&mut self, inputs: Array4<N>, policies: Array3<N>, values: Array1<N>) -> Result<(), Self::E> {
    Python::with_gil(|py| {
      let locals = PyDict::new(py);
      locals.set_item("torch", py.import("torch")?)?;
      locals.set_item("inputs", inputs.into_pyarray(py))?;
      locals.set_item("policies", policies.into_pyarray(py))?;
      locals.set_item("values", values.into_pyarray(py))?;
      locals.set_item("model", &self.model)?;
      locals.set_item("device", self.device.as_ref())?;

      // Need to create a new optimizer to make sure it points to correct model
      // after moving it to a different device. Probably it can be improved somehow...
      locals.set_item("old_optimizer", &self.optimizer)?;
      self.optimizer = py
        .eval(
          "torch.optim.AdamW(model.parameters(), weight_decay = 1e-4)",
          None,
          Some(locals),
        )?
        .extract()?;
      locals.set_item("optimizer", &self.optimizer)?;
      py.run(
        "optimizer.load_state_dict(old_optimizer.state_dict())",
        None,
        Some(locals),
      )?;

      py.run("model.train()", None, Some(locals))?;
      py.run(
        "model.train_on(optimizer, torch.from_numpy(inputs).to(device), torch.from_numpy(policies).to(device), torch.from_numpy(values).to(device))",
        None,
        Some(locals),
      )?;

      Ok(())
    })
  }

  fn save(&self) -> Result<(), Self::E> {
    PyModel::<N>::transfer(&self.model, "cpu")?;

    Python::with_gil(|py| {
      let locals = PyDict::new(py);
      locals.set_item("torch", py.import("torch")?)?;
      locals.set_item("model", &self.model)?;
      locals.set_item("optimizer", &self.optimizer)?;
      locals.set_item("path", self.path.as_ref())?;

      py.run(
        "torch.save({ 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict() }, path)",
        None,
        Some(locals),
      )
    })?;

    PyModel::<N>::transfer(&self.model, self.device.as_ref())
  }
}

impl<N: DType> Clone for PyModel<N> {
  fn clone(&self) -> Self {
    self.try_clone().unwrap()
  }
}
