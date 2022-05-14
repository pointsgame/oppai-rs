{
  inputs = {
    nixpkgs = {
      type = "github";
      owner = "NixOS";
      repo = "nixpkgs";
      ref = "nixos-21.11";
    };

    fenix = {
      type = "github";
      owner = "nix-community";
      repo = "fenix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs = inputs:
    let
      pkgs = import inputs.nixpkgs {
        system = "x86_64-linux";
        overlays = [ inputs.fenix.overlay ];
      };
    in {
      devShell.x86_64-linux = pkgs.mkShell {
        nativeBuildInputs = with pkgs; [
          cmake
          pkg-config
          (fenix.stable.withComponents [
            "cargo"
            "clippy"
            "rust-src"
            "rustc"
            "rustfmt"
          ])
          rust-analyzer
        ];

        buildInputs = with pkgs; [
          atk
          cairo
          freetype
          gdk-pixbuf
          glib
          gtk3
          pango
          vulkan-loader
        ];

        LD_LIBRARY_PATH = "${pkgs.vulkan-loader}/lib";
      };
    };
}
