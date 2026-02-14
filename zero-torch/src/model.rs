use std::borrow::Cow;
use std::ffi::{CStr, CString};
use std::marker::PhantomData;
use std::path::PathBuf;
use std::sync::Arc;

use indoc::indoc;
use ndarray::{Array1, Array3, Array4};
use num_traits::Float;
use numpy::array::{PyArray1, PyArray3};
use numpy::{Element, IntoPyArray, PyArrayMethods};
use oppai_zero::model::{Model, TrainableModel};
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyDict};

const MODEL_PY: &str = include_str!("../model.py");

pub trait DType {
  fn dtype() -> &'static CStr;
}

impl DType for f32 {
  fn dtype() -> &'static CStr {
    c"torch.float32"
  }
}

impl DType for f64 {
  fn dtype() -> &'static CStr {
    c"torch.float64"
  }
}

pub struct PyModel<N> {
  phantom: PhantomData<N>,
  device: Arc<Cow<'static, str>>,
  model: Py<PyAny>,
  optimizer: Py<PyAny>,
  lr: f64,
}

impl<N: DType> PyModel<N> {
  pub fn new(channels: u32, lr: f64) -> PyResult<Self> {
    Python::attach(|py| {
      let model = PyModule::from_code(py, CString::new(MODEL_PY).unwrap().as_c_str(), c"model.py", c"model")?;
      let locals = [("torch", py.import("torch")?), ("model", model)].into_py_dict(py)?;
      locals.set_item("channels", channels)?;
      locals.set_item("lr", lr)?;
      let dtype = py.eval(N::dtype(), None, Some(&locals))?;
      locals.set_item("dtype", dtype)?;
      let model: Py<PyAny> = py.eval(c"model.Model(channels).to(dtype)", None, Some(&locals))?.into();
      locals.set_item("model", &model)?;
      let optimizer: Py<PyAny> = py
        .eval(
          c"torch.optim.SGD(model.parameters(), lr, weight_decay = 1e-4)",
          None,
          Some(&locals),
        )?
        .into();

      Ok(Self {
        phantom: PhantomData,
        device: Arc::new(Cow::Borrowed("cpu")),
        model,
        optimizer,
        lr,
      })
    })
  }

  pub fn load(&self, path: PathBuf) -> PyResult<()> {
    PyModel::<N>::transfer(&self.model, "cpu")?;

    Python::attach(|py| {
      let locals = PyDict::new(py);
      locals.set_item("torch", py.import("torch")?)?;
      locals.set_item("model", &self.model)?;
      locals.set_item("optimizer", &self.optimizer)?;
      locals.set_item("path", path)?;

      py.run(
        indoc! {c"
          checkpoint = torch.load(path)
          model.load_state_dict(checkpoint['model_state_dict'])
          optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        "},
        None,
        Some(&locals),
      )
    })?;

    PyModel::<N>::transfer(&self.model, self.device.as_ref())
  }

  pub fn save(&self, path: PathBuf) -> PyResult<()> {
    PyModel::<N>::transfer(&self.model, "cpu")?;

    Python::attach(|py| {
      let locals = PyDict::new(py);
      locals.set_item("torch", py.import("torch")?)?;
      locals.set_item("model", &self.model)?;
      locals.set_item("optimizer", &self.optimizer)?;
      locals.set_item("path", path)?;

      py.run(
        c"torch.save({ 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict() }, path)",
        None,
        Some(&locals),
      )
    })?;

    PyModel::<N>::transfer(&self.model, self.device.as_ref())
  }

  pub fn try_clone(&self) -> PyResult<Self> {
    PyModel::<N>::transfer(&self.model, "cpu")?;

    let result = Python::attach(|py| -> PyResult<PyModel<N>> {
      let locals = PyDict::new(py);
      locals.set_item("copy", py.import("copy")?)?;
      locals.set_item("torch", py.import("torch")?)?;
      locals.set_item("model", &self.model)?;
      locals.set_item("lr", self.lr)?;

      let model: Py<PyAny> = py.eval(c"copy.deepcopy(model)", None, Some(&locals))?.into();

      locals.set_item("model", &model)?;
      let optimizer: Py<PyAny> = py
        .eval(
          c"torch.optim.SGD(model.parameters(), lr, weight_decay = 1e-4)",
          None,
          Some(&locals),
        )?
        .into();

      locals.set_item("old_optimizer", &self.optimizer)?;
      locals.set_item("new_optimizer", &optimizer)?;
      py.run(
        c"new_optimizer.load_state_dict(old_optimizer.state_dict())",
        None,
        Some(&locals),
      )?;

      Ok(Self {
        phantom: self.phantom,
        device: self.device.clone(),
        model,
        optimizer,
        lr: self.lr,
      })
    })?;

    PyModel::<N>::transfer(&self.model, self.device.as_ref())?;
    PyModel::<N>::transfer(&result.model, result.device.as_ref())?;

    Ok(result)
  }

  fn transfer(model: &Py<PyAny>, device: &str) -> PyResult<()> {
    Python::attach(|py| {
      let locals = PyDict::new(py);
      locals.set_item("model", model)?;
      locals.set_item("device", device)?;

      py.run(c"model.to(device)", None, Some(&locals))
    })
  }

  pub fn to_device(&mut self, device: Cow<'static, str>) -> PyResult<()> {
    self.device = Arc::new(device);

    PyModel::<N>::transfer(&self.model, self.device.as_ref())
  }
}

impl<N: Float + Element + DType> Model<N> for PyModel<N> {
  type E = PyErr;

  fn predict(&mut self, inputs: Array4<N>) -> Result<(Array3<N>, Array1<N>), Self::E> {
    Python::attach(|py| {
      let locals = PyDict::new(py);
      locals.set_item("torch", py.import("torch")?)?;
      locals.set_item("inputs", inputs.into_pyarray(py))?;
      locals.set_item("model", &self.model)?;
      locals.set_item("device", self.device.as_ref())?;

      py.run(c"model.eval()", None, Some(&locals))?;
      py.run(
        c"policies, values = map(lambda x : x.detach().cpu().numpy(), model.predict(torch.from_numpy(inputs).to(device)))",
        None,
        Some(&locals),
      )?;

      let policies: Bound<PyArray3<N>> = locals.get_item("policies")?.unwrap().cast_into().unwrap();
      let values: Bound<PyArray1<N>> = locals.get_item("values")?.unwrap().cast_into().unwrap();

      Ok((
        policies.try_readonly().unwrap().as_array().to_owned(),
        values.readonly().as_array().to_owned(),
      ))
    })
  }
}

impl<N: Float + Element + DType> TrainableModel<N> for PyModel<N> {
  type TE = Self::E;

  fn train(mut self, inputs: Array4<N>, policies: Array3<N>, values: Array1<N>) -> Result<Self, Self::E> {
    Python::attach(|py| {
      let locals = PyDict::new(py);
      locals.set_item("torch", py.import("torch")?)?;
      locals.set_item("inputs", inputs.into_pyarray(py))?;
      locals.set_item("policies", policies.into_pyarray(py))?;
      locals.set_item("values", values.into_pyarray(py))?;
      locals.set_item("model", &self.model)?;
      locals.set_item("device", self.device.as_ref())?;
      locals.set_item("lr", self.lr)?;

      // Need to create a new optimizer to make sure it points to correct model
      // after moving it to a different device. Probably it can be improved somehow...
      locals.set_item("old_optimizer", &self.optimizer)?;
      self.optimizer = py
        .eval(
          c"torch.optim.SGD(model.parameters(), lr, weight_decay = 1e-4)",
          None,
          Some(&locals),
        )?
        .into();
      locals.set_item("optimizer", &self.optimizer)?;
      py.run(
        c"optimizer.load_state_dict(old_optimizer.state_dict())",
        None,
        Some(&locals),
      )?;

      py.run(c"model.train()", None, Some(&locals))?;
      py.run(
        c"model.train_on(optimizer, torch.from_numpy(inputs).to(device), torch.from_numpy(policies).to(device), torch.from_numpy(values).to(device))",
        None,
        Some(&locals),
      )?;

      Ok(self)
    })
  }
}

impl<N: DType> Clone for PyModel<N> {
  fn clone(&self) -> Self {
    self.try_clone().unwrap()
  }
}
