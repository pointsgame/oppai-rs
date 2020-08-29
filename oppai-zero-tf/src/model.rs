use indoc::indoc;
use ndarray::{Array1, Array3, Array4};
use numpy::array::{PyArray1, PyArray3};
use numpy::IntoPyArray;
use oppai_zero::model::{Model, TrainableModel};
use pyo3::types::IntoPyDict;
use pyo3::{PyErr, PyObject, PyResult, Python};

pub struct PyModel<'a> {
  py: Python<'a>,
  model: PyObject,
}

impl<'a> PyModel<'a> {
  pub fn new(py: Python<'a>) -> PyResult<Self> {
    let locals = [("tf", py.import("tensorflow")?), ("np", py.import("numpy")?)].into_py_dict(py);
    let model: PyObject = py
      .eval("tf.keras.models.load_model('model.tf')", None, Some(&locals))?
      .extract()?;

    Ok(Self { py, model })
  }
}

impl<'a> Model for PyModel<'a> {
  type E = PyErr;

  fn predict(&self, inputs: Array4<f64>) -> Result<(Array3<f64>, Array1<f64>), Self::E> {
    let locals = [("tf", self.py.import("tensorflow")?), ("np", self.py.import("numpy")?)].into_py_dict(self.py);

    locals.set_item("inputs", inputs.into_pyarray(self.py))?;
    locals.set_item("model", &self.model)?;

    self.py.run(
      indoc!(
        "
          outputs = model.predict(inputs)
          policies = outputs[0]
          values = outputs[1]
        "
      ),
      None,
      Some(&locals),
    )?;

    let policies: &PyArray3<f64> = locals.get_item("policies").unwrap().extract()?;
    let values: &PyArray1<f64> = locals.get_item("values").unwrap().extract()?;

    Ok((
      policies.readonly().as_array().to_owned(),
      values.readonly().as_array().to_owned(),
    ))
  }
}

impl<'a> TrainableModel for PyModel<'a> {
  fn train(&self, inputs: Array4<f64>, policies: Array3<f64>, values: Array1<f64>) -> Result<(), Self::E> {
    let locals = [("tf", self.py.import("tensorflow")?), ("np", self.py.import("numpy")?)].into_py_dict(self.py);

    locals.set_item("inputs", inputs.into_pyarray(self.py))?;
    locals.set_item("policies", policies.into_pyarray(self.py))?;
    locals.set_item("values", values.into_pyarray(self.py))?;
    locals.set_item("model", &self.model)?;

    self.py.run(
      "model.fit(inputs, {'policy_output': policies, 'value_output': values})",
      None,
      Some(&locals),
    )?;

    Ok(())
  }
}
