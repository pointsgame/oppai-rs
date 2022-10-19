use std::path::PathBuf;
use std::rc::Rc;

use indoc::indoc;
use ndarray::{Array1, Array3, Array4};
use numpy::array::{PyArray1, PyArray3};
use numpy::IntoPyArray;
use oppai_zero::model::{Model, TrainableModel};
use pyo3::prelude::*;
use pyo3::types::{IntoPyDict, PyDict};

const OPPAI_NET: &str = include_str!("../oppai_net.py");

pub struct PyModel {
  path: Rc<PathBuf>,
  model: PyObject,
  optimizer: PyObject,
}

impl PyModel {
  pub fn new(path: PathBuf, width: u32, height: u32, channels: u32) -> PyResult<Self> {
    Python::with_gil(|py| {
      let oppai_net = PyModule::from_code(py, OPPAI_NET, "oppai_net.py", "oppai_net")?;
      let locals = [("torch", py.import("torch")?), ("oppai_net", oppai_net)].into_py_dict(py);
      locals.set_item("width", width.into_py(py))?;
      locals.set_item("height", height.into_py(py))?;
      locals.set_item("channels", channels.into_py(py))?;
      let model: PyObject = py
        .eval(
          "oppai_net.OppaiNet(width, height, channels).double()",
          None,
          Some(locals),
        )?
        .extract()?;
      locals.set_item("model", &model)?;
      let optimizer: PyObject = py
        .eval("torch.optim.Adam(model.parameters())", None, Some(locals))?
        .extract()?;

      Ok(Self {
        path: Rc::new(path),
        model,
        optimizer,
      })
    })
  }

  pub fn load(&self) -> PyResult<()> {
    Python::with_gil(|py| {
      let locals = PyDict::new(py);
      locals.set_item("torch", py.import("torch")?)?;
      locals.set_item("model", &self.model)?;
      locals.set_item("optimizer", &self.optimizer)?;
      locals.set_item("path", self.path.as_ref().into_py(py))?;

      py.run(
        indoc! {"
          checkpoint = torch.load(path)
          model.load_state_dict(checkpoint['model_state_dict'])
          optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        "},
        None,
        Some(locals),
      )
    })
  }

  pub fn try_clone(&self) -> PyResult<Self> {
    Python::with_gil(|py| {
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
        path: self.path.clone(),
        model,
        optimizer,
      })
    })
  }
}

impl Model for PyModel {
  type E = PyErr;

  fn predict(&self, inputs: Array4<f64>) -> Result<(Array3<f64>, Array1<f64>), Self::E> {
    Python::with_gil(|py| {
      let locals = PyDict::new(py);
      locals.set_item("torch", py.import("torch")?)?;
      locals.set_item("inputs", inputs.into_pyarray(py))?;
      locals.set_item("model", &self.model)?;

      py.run("model.eval()", None, Some(locals))?;
      py.run(
        "policies, values = map(lambda x : x.detach().numpy(), model.predict(torch.from_numpy(inputs)))",
        None,
        Some(locals),
      )?;

      let policies: &PyArray3<f64> = locals.get_item("policies").unwrap().extract()?;
      let values: &PyArray1<f64> = locals.get_item("values").unwrap().extract()?;

      Ok((
        policies.readonly().as_array().to_owned(),
        values.readonly().as_array().to_owned(),
      ))
    })
  }
}

impl TrainableModel for PyModel {
  fn train(&self, inputs: Array4<f64>, policies: Array3<f64>, values: Array1<f64>) -> Result<(), Self::E> {
    Python::with_gil(|py| {
      let locals = PyDict::new(py);
      locals.set_item("torch", py.import("torch")?)?;
      locals.set_item("inputs", inputs.into_pyarray(py))?;
      locals.set_item("policies", policies.into_pyarray(py))?;
      locals.set_item("values", values.into_pyarray(py))?;
      locals.set_item("model", &self.model)?;
      locals.set_item("optimizer", &self.optimizer)?;

      py.run("model.train()", None, Some(locals))?;
      py.run(
        "model.train_on(optimizer, torch.from_numpy(inputs), torch.from_numpy(policies), torch.from_numpy(values))",
        None,
        Some(locals),
      )?;

      Ok(())
    })
  }

  fn save(&self) -> Result<(), Self::E> {
    Python::with_gil(|py| {
      let locals = PyDict::new(py);
      locals.set_item("torch", py.import("torch")?)?;
      locals.set_item("model", &self.model)?;
      locals.set_item("optimizer", &self.optimizer)?;
      locals.set_item("path", self.path.as_ref().into_py(py))?;

      py.run(
        "torch.save({ 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict() }, path)",
        None,
        Some(locals),
      )
    })
  }
}

impl Clone for PyModel {
  fn clone(&self) -> Self {
    self.try_clone().unwrap()
  }
}
