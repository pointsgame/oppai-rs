{
  inputs = {
    nixpkgs = {
      type = "github";
      owner = "NixOS";
      repo = "nixpkgs";
      ref = "nixos-unstable";
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
        overlays = [
          inputs.fenix.overlay
          (self: super: {
            python3 = super.python3.override {
              packageOverrides = python-self: python-super: {
                pytorch = python-super.pytorch.overrideAttrs (old: {
                  patches = old.patches ++ [
                    (pkgs.fetchpatch {
                      url =
                        "https://github.com/pytorch/pytorch/commit/eb74af18af6e90ae47f24997af8468bf7b9deb72.patch";
                      sha256 =
                        "sha256-IRmBW05naNEWBttiRfURm6jGd7UpsYpNiwqxkiGn7l4=";
                    })
                  ];
                });
              };
            };
            python3Packages = self.python3.pkgs;
          })
        ];
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

        buildInputs = let
          dlprimitives = (pkgs.callPackage ./dlprimitives.nix { });
          pytorch-dlprim = (pkgs.callPackage ./pytorch-dlprim.nix {
            dlprimitives = dlprimitives;
          });
        in with pkgs; [
          atk
          cairo
          freetype
          gdk-pixbuf
          glib
          gtk3
          pango
          vulkan-loader
          ((python3.withPackages
            (pkgs: with pkgs; [ pytorch torchvision ])).override
            (args: { ignoreCollisions = true; }))
          dlprimitives
          pytorch-dlprim
        ];

        LD_LIBRARY_PATH = "${pkgs.vulkan-loader}/lib";
      };
    };
}
