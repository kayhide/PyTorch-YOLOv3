let
  nixpkgs-src = builtins.fetchTarball {
    # master of 2021-01-05.
    url = "https://github.com/NixOS/nixpkgs/archive/d9dba88d08a9cdf483c3d45f0d7220cf97a4ce64.tar.gz";
    sha256 = "1ww9w7pkrr2jfszln5ifsrn200phdzn7ppf0p872wg0yfgrdpk2c";
  };

  nixpkgs = import nixpkgs-src { config = { allowUnfree = true; }; };

  inherit (nixpkgs) pkgs lib stdenv;

  lib-path = lib.makeLibraryPath [
    stdenv.cc.cc
    pkgs.linuxPackages.nvidia_x11
  ];

in

pkgs.mkShell {
  buildInputs = with pkgs; [
    unzip
    python3
  ];

  shellHook = ''
    export "LD_LIBRARY_PATH=${lib-path}"
  '';
}
