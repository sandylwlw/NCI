declare module "isosurface" {
  type Vec3 = [number, number, number];
  type Mesh = { positions: Vec3[]; cells: [number, number, number][] };

  const isosurface: {
    marchingCubes(
      dims: [number, number, number],
      field: (x: number, y: number, z: number) => number,
      level?: number
    ): Mesh;
  };

  export default isosurface;
}
