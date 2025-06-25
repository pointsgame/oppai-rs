use std::io::{Error, Result};

use async_process::{Child, ChildStdin, ChildStdout, Command, Stdio};
use futures::{
  AsyncBufReadExt,
  io::{AsyncWriteExt, BufReader},
};
use log::debug;
use oppai_field::player::Player;
pub use oppai_protocol::Constraint;
use oppai_protocol::{Coords, Move, Request, Response};

pub struct Client {
  _child: Child,
  stdin: ChildStdin,
  stdout: BufReader<ChildStdout>,
}

impl Client {
  pub fn spawn<I: IntoIterator<Item = String>>(program: String, args: I) -> Result<Self> {
    let mut child = Command::new(program)
      .args(args)
      .stdin(Stdio::piped())
      .stdout(Stdio::piped())
      .kill_on_drop(true)
      .spawn()?;
    let stdin = child.stdin.take().ok_or(Error::other("No stdin"))?;
    let stdout = BufReader::new(child.stdout.take().ok_or(Error::other("No stdout"))?);

    Ok(Client {
      _child: child,
      stdin,
      stdout,
    })
  }

  async fn request(&mut self, request: Request) -> Result<()> {
    debug!("Request: {:?}", request);
    let mut bytes = serde_json::to_vec(&request)?;
    bytes.push(b"\n"[0]);
    self.stdin.write_all(bytes.as_slice()).await
  }

  async fn response(&mut self) -> Result<Response> {
    let mut s = String::new();
    self.stdout.read_line(&mut s).await?;
    let response = serde_json::from_str::<Response>(&s)?;
    debug!("Response: {:?}", response);
    Ok(response)
  }

  pub async fn init(&mut self, width: u32, height: u32) -> Result<()> {
    self.request(Request::Init { width, height }).await?;

    let response = self.response().await?;
    if let Response::Init = response {
      Ok(())
    } else {
      Err(Error::other(format!("Wrong response type: {:?}", response)))
    }
  }

  pub async fn put_point(&mut self, x: u32, y: u32, player: Player) -> Result<bool> {
    self
      .request(Request::PutPoint {
        coords: Coords { x, y },
        player,
      })
      .await?;

    let response = self.response().await?;

    if let Response::PutPoint { put } = response {
      Ok(put)
    } else {
      Err(Error::other(format!("Wrong response type: {:?}", response)))
    }
  }

  pub async fn undo(&mut self) -> Result<bool> {
    self.request(Request::Undo).await?;

    let response = self.response().await?;

    if let Response::Undo { undone } = response {
      Ok(undone)
    } else {
      Err(Error::other(format!("Wrong response type: {:?}", response)))
    }
  }

  pub async fn analyze(&mut self, player: Player, constraint: Constraint) -> Result<Vec<Move>> {
    self.request(Request::Analyze { player, constraint }).await?;

    let response = self.response().await?;

    if let Response::Analyze { moves } = response {
      Ok(moves)
    } else {
      Err(Error::other(format!("Wrong response type: {:?}", response)))
    }
  }
}
