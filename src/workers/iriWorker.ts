import isosurface from "isosurface";

type WorkerRequest = {
  type: "compute";
  payload: {
    fchkText: string;
    gridSize: number;
    padding: number;
    aValue: number;
    isoValue: number;
    colorRange: { min: number; max: number };
  };
};

type WorkerProgress = {
  type: "progress";
  stage: string;
  percent: number;
};

type WorkerError = {
  type: "error";
  message: string;
};

type WorkerResult = {
  type: "result";
  payload: {
    vertices: Float32Array;
    colors: Float32Array;
    indices: Uint32Array;
    bounds: { min: [number, number, number]; max: [number, number, number] };
    stats: {
      gridSize: number;
      basisCount: number;
      occOrbitals: number;
      maxDensity: number;
    };
  };
};

type ParsedFchk = {
  atoms: { atomicNumber: number; x: number; y: number; z: number }[];
  shellTypes: number[];
  shellToAtom: number[];
  nPrimsPerShell: number[];
  primExponents: number[];
  contractionCoeffs: number[];
  spContractionCoeffs: number[] | null;
  alphaMOCoeffs: number[];
  nAlpha: number;
  nBeta: number;
  nBasis: number;
  pureCartesianD: number | null;
  pureCartesianF: number | null;
};

type BasisFunction =
  | {
      kind: "cartesian";
      center: [number, number, number];
      lx: number;
      ly: number;
      lz: number;
      primitives: { exp: number; coeff: number }[];
    }
  | {
      kind: "spherical";
      center: [number, number, number];
      l: number;
      component: { m: number; kind: "c" | "s" | "0" };
      primitives: { exp: number; coeff: number }[];
    };

const CARTESIAN_LMN: Record<number, Array<[number, number, number]>> = {
  0: [[0, 0, 0]],
  1: [
    [1, 0, 0],
    [0, 1, 0],
    [0, 0, 1],
  ],
  2: [
    [2, 0, 0],
    [0, 2, 0],
    [0, 0, 2],
    [1, 1, 0],
    [1, 0, 1],
    [0, 1, 1],
  ],
  3: [
    [3, 0, 0],
    [0, 3, 0],
    [0, 0, 3],
    [2, 1, 0],
    [2, 0, 1],
    [1, 2, 0],
    [0, 2, 1],
    [1, 0, 2],
    [0, 1, 2],
    [1, 1, 1],
  ],
  4: [
    [4, 0, 0],
    [0, 4, 0],
    [0, 0, 4],
    [3, 1, 0],
    [3, 0, 1],
    [1, 3, 0],
    [0, 3, 1],
    [1, 0, 3],
    [0, 1, 3],
    [2, 2, 0],
    [2, 0, 2],
    [0, 2, 2],
    [2, 1, 1],
    [1, 2, 1],
    [1, 1, 2],
  ],
};

const SPHERICAL_COMPONENTS: Record<
  number,
  Array<{ m: number; kind: "c" | "s" | "0" }>
> = {
  2: [
    { m: 0, kind: "0" },
    { m: 1, kind: "c" },
    { m: 1, kind: "s" },
    { m: 2, kind: "c" },
    { m: 2, kind: "s" },
  ],
  3: [
    { m: 0, kind: "0" },
    { m: 1, kind: "c" },
    { m: 1, kind: "s" },
    { m: 2, kind: "c" },
    { m: 2, kind: "s" },
    { m: 3, kind: "c" },
    { m: 3, kind: "s" },
  ],
  4: [
    { m: 0, kind: "0" },
    { m: 1, kind: "c" },
    { m: 1, kind: "s" },
    { m: 2, kind: "c" },
    { m: 2, kind: "s" },
    { m: 3, kind: "c" },
    { m: 3, kind: "s" },
    { m: 4, kind: "c" },
    { m: 4, kind: "s" },
  ],
};

const EPS = 1e-12;

self.onmessage = (event: MessageEvent<WorkerRequest>) => {
  if (event.data.type !== "compute") return;
  const { fchkText, gridSize, padding, aValue, isoValue, colorRange } =
    event.data.payload;
  let currentStage = "start";

  try {
    currentStage = "parse";
    postProgress("parse", 0);
    const parsed = parseFchk(fchkText);
    if (parsed.nAlpha !== parsed.nBeta) {
      throw new Error(
        "Open-shell systems are not supported in this version."
      );
    }

    const resolved = resolveShellTypes(parsed.shellTypes, {
      d: parsed.pureCartesianD,
      f: parsed.pureCartesianF,
    }, parsed.nBasis);

    const basisCount = countBasisFunctions(resolved.shellTypes);
    if (basisCount !== parsed.nBasis) {
      throw new Error(
        `Basis count mismatch: expected ${parsed.nBasis}, got ${basisCount}`
      );
    }
    if (parsed.alphaMOCoeffs.length % basisCount !== 0) {
      throw new Error(
        `Alpha MO coefficients length mismatch: expected multiple of ${basisCount}, got ${parsed.alphaMOCoeffs.length}`
      );
    }

    currentStage = "basis";
    postProgress("basis", 0);
    const basisFunctions = buildBasisFunctions(parsed, resolved.shellTypes);

    currentStage = "grid";
    postProgress("grid", 0);
    const bounds = computeBounds(parsed.atoms, padding);
    const dims: [number, number, number] = [gridSize, gridSize, gridSize];
    const spacing: [number, number, number] = [
      (bounds.max[0] - bounds.min[0]) / (gridSize - 1),
      (bounds.max[1] - bounds.min[1]) / (gridSize - 1),
      (bounds.max[2] - bounds.min[2]) / (gridSize - 1),
    ];

    currentStage = "density";
    postProgress("density", 0);
    const { rho, maxDensity } = computeDensity(
      basisFunctions,
      parsed.alphaMOCoeffs,
      parsed.nAlpha,
      dims,
      bounds,
      spacing
    );

    currentStage = "derivatives";
    postProgress("derivatives", 0);
    const { iri, signLambda2Rho, minIri, maxIri, nanCount, infCount } = computeIriAndSign(
      rho,
      dims,
      spacing,
      aValue
    );
    if (nanCount > 0 || infCount > 0) {
      throw new Error(
        `IRI contains invalid values (NaN: ${nanCount}, Inf: ${infCount}). Try increasing padding or lowering a/isovalue. Range so far: [${minIri.toExponential(3)}, ${maxIri.toExponential(3)}]`
      );
    }

    currentStage = "surface";
    postProgress("surface", 0);
    const mesh = buildIsosurface(
      iri,
      signLambda2Rho,
      dims,
      bounds,
      spacing,
      isoValue,
      colorRange,
      { minIri, maxIri }
    );

    const result: WorkerResult = {
      type: "result",
      payload: {
        vertices: mesh.vertices,
        colors: mesh.colors,
        indices: mesh.indices,
        bounds,
        stats: {
          gridSize,
          basisCount: basisFunctions.length,
          occOrbitals: parsed.nAlpha,
          maxDensity,
        },
      },
    };
    (self as unknown as Worker).postMessage(result, [
      result.payload.vertices.buffer,
      result.payload.colors.buffer,
      result.payload.indices.buffer,
    ]);
  } catch (error) {
    const err = error instanceof Error ? error : new Error(String(error));
    const message = `${err.message}\nStage: ${currentStage}\n${err.stack ?? ""}`;
    const diag = formatLabelDiagnostics(
      fchkText.replace(/\r/g, "").split("\n")
    );
    const errPayload: WorkerError = { type: "error", message: `${message}\n${diag}` };
    (self as unknown as Worker).postMessage(errPayload);
  }
};

function postProgress(stage: string, percent: number) {
  const payload: WorkerProgress = { type: "progress", stage, percent };
  (self as unknown as Worker).postMessage(payload);
}

function parseFchk(text: string): ParsedFchk {
  const lines = text.replace(/\r/g, "").split("\n");

  const nAtoms = readScalar(lines, "Number of atoms");
  const nAlpha = readScalar(lines, "Number of alpha electrons");
  const nBeta = readScalar(lines, "Number of beta electrons");
  const nBasis = readScalar(lines, "Number of basis functions");
  const pureCartesianD = tryReadScalar(lines, "Pure/Cartesian d shells");
  const pureCartesianF = tryReadScalar(lines, "Pure/Cartesian f shells");

  const atomicNumbers = tryReadArray(lines, "Atomic numbers");
  const coords = readArray(lines, "Current cartesian coordinates");
  const shellTypes = readArray(lines, "Shell types");
  const shellToAtom = readArray(lines, "Shell to atom map");
  const nPrimsPerShell = readArray(lines, "Number of primitives per shell");
  const primExponents = readArray(lines, "Primitive exponents");
  const contractionCoeffs = readArray(lines, "Contraction coefficients");
  const spContractionCoeffs = tryReadArray(
    lines,
    "P(S=P) Contraction coefficients"
  );
  const alphaMOCoeffs = readArray(lines, "Alpha MO coefficients");

  const resolvedAtomicNumbers = atomicNumbers ?? tryReadArray(lines, "Nuclear charges");
  if (!resolvedAtomicNumbers) {
    throw new Error("Missing array: Atomic numbers");
  }
  if (resolvedAtomicNumbers.length !== nAtoms) {
    throw new Error("Failed to read atomic numbers from .fchk");
  }
  if (coords.length !== nAtoms * 3) {
    throw new Error("Failed to read coordinates from .fchk");
  }

  const atoms = resolvedAtomicNumbers.map((atomicNumber, i) => ({
    atomicNumber: Math.round(atomicNumber),
    x: coords[i * 3],
    y: coords[i * 3 + 1],
    z: coords[i * 3 + 2],
  }));

  return {
    atoms,
    shellTypes: shellTypes.map((v) => Math.trunc(v)),
    shellToAtom: shellToAtom.map((v) => Math.trunc(v)),
    nPrimsPerShell: nPrimsPerShell.map((v) => Math.trunc(v)),
    primExponents,
    contractionCoeffs,
    spContractionCoeffs,
    alphaMOCoeffs,
    nAlpha,
    nBeta,
    nBasis,
    pureCartesianD,
    pureCartesianF,
  };
}

function readScalar(lines: string[], label: string): number {
  const labelRegex = new RegExp(`^\\s*${escapeLabel(label)}\\b`, "i");
  for (const line of lines) {
    if (!labelRegex.test(line)) continue;
    const nMatch = line.match(/N=\s*(-?\d+)/i);
    if (nMatch) return Number(nMatch[1]);
    const tailMatch = line.match(/(-?\d+)\s*$/);
    if (tailMatch) return Number(tailMatch[1]);
  }
  throw new Error(`Missing scalar value: ${label}`);
}

function tryReadScalar(lines: string[], label: string): number | null {
  const labelRegex = new RegExp(`^\\s*${escapeLabel(label)}\\b`, "i");
  for (const line of lines) {
    if (!labelRegex.test(line)) continue;
    const nMatch = line.match(/N=\s*(-?\d+)/i);
    if (nMatch) return Number(nMatch[1]);
    const tailMatch = line.match(/(-?\d+)\s*$/);
    if (tailMatch) return Number(tailMatch[1]);
  }
  return null;
}

function readArray(lines: string[], label: string): number[] {
  const result = tryReadArray(lines, label);
  if (!result) throw new Error(`Missing array: ${label}`);
  return result;
}

function tryReadArray(lines: string[], label: string): number[] | null {
  const labelRegex = new RegExp(`^\\s*${escapeLabel(label)}\\b`, "i");
  for (let i = 0; i < lines.length; i += 1) {
    const line = lines[i];
    if (!labelRegex.test(line)) continue;
    const countMatch = line.match(/N=\s*(\d+)/i) ?? line.match(/\s(\d+)\s*$/);
    if (!countMatch) {
      throw new Error(`Missing count for array: ${label}`);
    }
    const count = Number(countMatch[1]);
    const values: number[] = [];
    for (let j = i + 1; j < lines.length && values.length < count; j += 1) {
      const tokens = lines[j].trim().split(/\s+/).filter(Boolean);
      for (const token of tokens) {
        if (values.length >= count) break;
        values.push(parseNumber(token));
      }
    }
    if (values.length !== count) {
      throw new Error(`Failed to read ${label} (${values.length}/${count})`);
    }
    return values;
  }
  return null;
}

function formatLabelDiagnostics(lines: string[]): string {
  const required = [
    "Number of atoms",
    "Atomic numbers",
    "Nuclear charges",
    "Current cartesian coordinates",
    "Shell types",
    "Shell to atom map",
    "Number of primitives per shell",
    "Primitive exponents",
    "Contraction coefficients",
    "Alpha MO coefficients",
  ];

  const results = required.map((label) => {
    const regex = new RegExp(`^\\s*${escapeLabel(label)}\\b`, "i");
    const found = lines.some((line) => regex.test(line));
    return `${label}: ${found ? "yes" : "no"}`;
  });

  return `Detected labels -> ${results.join(" | ")}`;
}

function parseNumber(value: string): number {
  return Number(value.replace(/D/g, "E"));
}

function escapeLabel(label: string): string {
  return label.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function buildBasisFunctions(
  parsed: ParsedFchk,
  shellTypes: number[]
): BasisFunction[] {
  const basisFunctions: BasisFunction[] = [];
  let primOffset = 0;

  for (let s = 0; s < shellTypes.length; s += 1) {
    const shellType = shellTypes[s];
    const atomIndex = parsed.shellToAtom[s] - 1;
    const nPrims = parsed.nPrimsPerShell[s];
    const center = parsed.atoms[atomIndex];
    const primitives = [] as { exp: number; coeff: number }[];
    const primitivesSp = [] as { exp: number; coeff: number }[];

    for (let p = 0; p < nPrims; p += 1) {
      const exp = parsed.primExponents[primOffset + p];
      const coeff = parsed.contractionCoeffs[primOffset + p];
      primitives.push({ exp, coeff });

      if (shellType === -1) {
        if (!parsed.spContractionCoeffs) {
          throw new Error("Missing SP contraction coefficients in .fchk");
        }
        const spCoeff = parsed.spContractionCoeffs[primOffset + p];
        primitivesSp.push({ exp, coeff: spCoeff });
      }
    }

    primOffset += nPrims;

    if (shellType === -1) {
      pushShell(basisFunctions, center, 0, primitives);
      pushShell(basisFunctions, center, 1, primitivesSp);
      continue;
    }

    if (shellType < 0) {
      const l = Math.abs(shellType);
      pushSphericalShell(basisFunctions, center, l, primitives);
      continue;
    }

    pushShell(basisFunctions, center, shellType, primitives);
  }

  return basisFunctions;
}

function pushShell(
  basisFunctions: BasisFunction[],
  center: { x: number; y: number; z: number },
  shellType: number,
  primitives: { exp: number; coeff: number }[]
) {
  const lmnList = CARTESIAN_LMN[shellType];
  if (!lmnList) {
    throw new Error(`Shell type ${shellType} not supported`);
  }

  for (const [lx, ly, lz] of lmnList) {
    const prims = primitives.map((primitive) => ({
      exp: primitive.exp,
      coeff: primitive.coeff * primitiveNormalization(primitive.exp, lx, ly, lz),
    }));

    basisFunctions.push({
      kind: "cartesian",
      center: [center.x, center.y, center.z],
      lx,
      ly,
      lz,
      primitives: prims,
    });
  }
}

function pushSphericalShell(
  basisFunctions: BasisFunction[],
  center: { x: number; y: number; z: number },
  l: number,
  primitives: { exp: number; coeff: number }[]
) {
  const components = SPHERICAL_COMPONENTS[l];
  if (!components) {
    throw new Error(`Spherical shell type -${l} not supported`);
  }

  for (const component of components) {
    const prims = primitives.map((primitive) => ({
      exp: primitive.exp,
      coeff: primitive.coeff * sphericalPrimitiveNormalization(primitive.exp, l),
    }));

    basisFunctions.push({
      kind: "spherical",
      center: [center.x, center.y, center.z],
      l,
      component,
      primitives: prims,
    });
  }
}

function normalizeShellTypes(
  shellTypes: number[],
  flags: { d: number | null; f: number | null }
): number[] {
  const normalized = shellTypes.slice();
  for (let i = 0; i < normalized.length; i += 1) {
    const value = normalized[i];
    if (value >= 0 || value === -1) continue;
    const l = Math.abs(value);
    if (l === 2 && flags.d === 0) {
      normalized[i] = l;
      continue;
    }
    if (l === 3 && flags.f === 0) {
      normalized[i] = l;
    }
  }
  return normalized;
}

function resolveShellTypes(
  shellTypes: number[],
  flags: { d: number | null; f: number | null },
  nBasis: number
): { shellTypes: number[] } {
  const rawCount = countBasisFunctions(shellTypes);
  const normalized = normalizeShellTypes(shellTypes, flags);
  const normalizedCount = countBasisFunctions(normalized);

  if (rawCount === nBasis && normalizedCount !== nBasis) {
    return { shellTypes };
  }
  if (normalizedCount === nBasis && rawCount !== nBasis) {
    return { shellTypes: normalized };
  }
  if (normalizedCount === nBasis && rawCount === nBasis) {
    return { shellTypes: normalized };
  }

  const counts = shellTypeCounts(shellTypes);
  const detail = Object.entries(counts)
    .sort((a, b) => Number(a[0]) - Number(b[0]))
    .map(([k, v]) => `${k}:${v}`)
    .join(", ");
  throw new Error(
    `Basis count mismatch: expected ${nBasis}, raw ${rawCount}, normalized ${normalizedCount} (shells ${detail})`
  );
}

function shellTypeCounts(shellTypes: number[]): Record<number, number> {
  const counts: Record<number, number> = {};
  for (const value of shellTypes) {
    counts[value] = (counts[value] ?? 0) + 1;
  }
  return counts;
}

function countBasisFunctions(shellTypes: number[]): number {
  let total = 0;
  for (const shellType of shellTypes) {
    if (shellType === -1) {
      total += 4;
      continue;
    }
    if (shellType < 0) {
      total += sphericalCount(Math.abs(shellType));
      continue;
    }
    total += cartesianCount(shellType);
  }
  return total;
}

function cartesianCount(l: number): number {
  return ((l + 1) * (l + 2)) / 2;
}

function sphericalCount(l: number): number {
  return 2 * l + 1;
}

function primitiveNormalization(exp: number, lx: number, ly: number, lz: number): number {
  const l = lx + ly + lz;
  const prefactor = Math.pow((2 * exp) / Math.PI, 0.75);
  const num = Math.pow(4 * exp, l);
  const denom = doubleFactorial(2 * lx - 1) * doubleFactorial(2 * ly - 1) * doubleFactorial(2 * lz - 1);
  return prefactor * Math.sqrt(num / denom);
}

function sphericalPrimitiveNormalization(exp: number, l: number): number {
  const prefactor = Math.pow((2 * exp) / Math.PI, 0.75);
  const num = Math.pow(4 * exp, l);
  const denom = doubleFactorial(2 * l - 1);
  return prefactor * Math.sqrt(num / denom);
}

function doubleFactorial(n: number): number {
  if (n <= 0) return 1;
  let result = 1;
  for (let i = n; i > 1; i -= 2) result *= i;
  return result;
}

function computeBounds(
  atoms: { x: number; y: number; z: number }[],
  padding: number
) {
  let minX = Infinity;
  let minY = Infinity;
  let minZ = Infinity;
  let maxX = -Infinity;
  let maxY = -Infinity;
  let maxZ = -Infinity;

  for (const atom of atoms) {
    minX = Math.min(minX, atom.x);
    minY = Math.min(minY, atom.y);
    minZ = Math.min(minZ, atom.z);
    maxX = Math.max(maxX, atom.x);
    maxY = Math.max(maxY, atom.y);
    maxZ = Math.max(maxZ, atom.z);
  }

  return {
    min: [minX - padding, minY - padding, minZ - padding] as [number, number, number],
    max: [maxX + padding, maxY + padding, maxZ + padding] as [number, number, number],
  };
}

function computeDensity(
  basisFunctions: BasisFunction[],
  moCoeffs: number[],
  occOrbitals: number,
  dims: [number, number, number],
  bounds: { min: [number, number, number]; max: [number, number, number] },
  spacing: [number, number, number]
): { rho: Float32Array; maxDensity: number } {
  const [nx, ny, nz] = dims;
  const total = nx * ny * nz;
  const rho = new Float32Array(total);
  const basisCount = basisFunctions.length;
  const basisValues = new Float32Array(basisCount);
  let maxDensity = 0;

  for (let z = 0; z < nz; z += 1) {
    const zCoord = bounds.min[2] + z * spacing[2];
    if (z % 4 === 0) {
      postProgress("density", Math.round((z / (nz - 1)) * 100));
    }
    for (let y = 0; y < ny; y += 1) {
      const yCoord = bounds.min[1] + y * spacing[1];
      for (let x = 0; x < nx; x += 1) {
        const xCoord = bounds.min[0] + x * spacing[0];
        for (let i = 0; i < basisCount; i += 1) {
          basisValues[i] = evalBasis(basisFunctions[i], xCoord, yCoord, zCoord);
        }
        let density = 0;
        for (let mo = 0; mo < occOrbitals; mo += 1) {
          let moValue = 0;
          const offset = mo * basisCount;
          for (let i = 0; i < basisCount; i += 1) {
            moValue += moCoeffs[offset + i] * basisValues[i];
          }
          density += moValue * moValue;
        }
        density *= 2;
        const idx = x + y * nx + z * nx * ny;
        rho[idx] = density;
        if (density > maxDensity) maxDensity = density;
      }
    }
  }

  return { rho, maxDensity };
}

function evalBasis(fn: BasisFunction, x: number, y: number, z: number): number {
  const dx = x - fn.center[0];
  const dy = y - fn.center[1];
  const dz = z - fn.center[2];
  const r2 = dx * dx + dy * dy + dz * dz;

  let radial = 0;
  for (const primitive of fn.primitives) {
    radial += primitive.coeff * Math.exp(-primitive.exp * r2);
  }

  if (fn.kind === "cartesian") {
    const xPow = powByDegree(dx, fn.lx);
    const yPow = powByDegree(dy, fn.ly);
    const zPow = powByDegree(dz, fn.lz);
    return radial * xPow * yPow * zPow;
  }

  const angular = evalSolidHarmonic(fn.l, fn.component, dx, dy, dz);
  return radial * angular;
}

function evalSolidHarmonic(
  l: number,
  component: { m: number; kind: "c" | "s" | "0" },
  x: number,
  y: number,
  z: number
): number {
  const x2 = x * x;
  const y2 = y * y;
  const z2 = z * z;
  const r2 = x2 + y2 + z2;
  const { m, kind } = component;

  if (l === 2) {
    const c20 = Math.sqrt(5 / (16 * Math.PI));
    const c21 = Math.sqrt(15 / (4 * Math.PI));
    const c22 = Math.sqrt(15 / (16 * Math.PI));
    if (m === 0) return c20 * (2 * z2 - x2 - y2);
    if (m === 1 && kind === "c") return c21 * x * z;
    if (m === 1 && kind === "s") return c21 * y * z;
    if (m === 2 && kind === "c") return c22 * (x2 - y2);
    if (m === 2 && kind === "s") return (2 * c22) * x * y;
  }

  if (l === 3) {
    const c30 = Math.sqrt(7 / (16 * Math.PI));
    const c31 = Math.sqrt(21 / (32 * Math.PI));
    const c32 = Math.sqrt(105 / (16 * Math.PI));
    const c33 = Math.sqrt(35 / (32 * Math.PI));
    if (m === 0) return c30 * (2 * z * z2 - 3 * z * (x2 + y2));
    if (m === 1 && kind === "c") return c31 * x * (5 * z2 - r2);
    if (m === 1 && kind === "s") return c31 * y * (5 * z2 - r2);
    if (m === 2 && kind === "c") return c32 * z * (x2 - y2);
    if (m === 2 && kind === "s") return (2 * c32) * x * y * z;
    if (m === 3 && kind === "c") return c33 * (x * (x2 - 3 * y2));
    if (m === 3 && kind === "s") return c33 * (y * (3 * x2 - y2));
  }

  if (l === 4) {
    const c40 = (3 / 16) * Math.sqrt(1 / Math.PI);
    const c41 = (3 / 4) * Math.sqrt(5 / Math.PI);
    const c42 = (3 / 8) * Math.sqrt(5 / Math.PI);
    const c43 = (3 / 8) * Math.sqrt(70 / Math.PI);
    const c44 = (3 / 16) * Math.sqrt(35 / Math.PI);
    const r4 = r2 * r2;
    if (m === 0) return c40 * (35 * z2 * z2 - 30 * z2 * r2 + 3 * r4);
    if (m === 1 && kind === "c") return c41 * x * z * (7 * z2 - 3 * r2);
    if (m === 1 && kind === "s") return c41 * y * z * (7 * z2 - 3 * r2);
    if (m === 2 && kind === "c") return c42 * (x2 - y2) * (7 * z2 - r2);
    if (m === 2 && kind === "s") return (2 * c42) * x * y * (7 * z2 - r2);
    if (m === 3 && kind === "c") return c43 * z * (x * (x2 - 3 * y2));
    if (m === 3 && kind === "s") return c43 * z * (y * (3 * x2 - y2));
    if (m === 4 && kind === "c") return c44 * (x2 * x2 - 6 * x2 * y2 + y2 * y2);
    if (m === 4 && kind === "s") return c44 * (4 * x * y * (x2 - y2));
  }

  throw new Error(`Spherical harmonic l=${l} m=${m}${kind} not supported`);
}

function powByDegree(value: number, degree: number): number {
  if (degree === 0) return 1;
  if (degree === 1) return value;
  if (degree === 2) return value * value;
  if (degree === 3) return value * value * value;
  if (degree === 4) return value * value * value * value;
  return Math.pow(value, degree);
}

function computeIriAndSign(
  rho: Float32Array,
  dims: [number, number, number],
  spacing: [number, number, number],
  aValue: number
) {
  const [nx, ny, nz] = dims;
  const total = nx * ny * nz;
  const iri = new Float32Array(total);
  const signLambda2Rho = new Float32Array(total);
  let minIri = Infinity;
  let maxIri = -Infinity;
  let nanCount = 0;
  let infCount = 0;
  const hx = spacing[0];
  const hy = spacing[1];
  const hz = spacing[2];
  const hx2 = hx * hx;
  const hy2 = hy * hy;
  const hz2 = hz * hz;

  for (let z = 0; z < nz; z += 1) {
    if (z % 4 === 0) {
      postProgress("derivatives", Math.round((z / (nz - 1)) * 100));
    }
    for (let y = 0; y < ny; y += 1) {
      for (let x = 0; x < nx; x += 1) {
        const idx = x + y * nx + z * nx * ny;
        const rhoC = rho[idx];
        if (rhoC < EPS) {
          iri[idx] = 0;
          signLambda2Rho[idx] = 0;
          continue;
        }

        const xm = x > 0 ? x - 1 : x;
        const xp = x < nx - 1 ? x + 1 : x;
        const ym = y > 0 ? y - 1 : y;
        const yp = y < ny - 1 ? y + 1 : y;
        const zm = z > 0 ? z - 1 : z;
        const zp = z < nz - 1 ? z + 1 : z;

        const rhoXp = rho[xp + y * nx + z * nx * ny];
        const rhoXm = rho[xm + y * nx + z * nx * ny];
        const rhoYp = rho[x + yp * nx + z * nx * ny];
        const rhoYm = rho[x + ym * nx + z * nx * ny];
        const rhoZp = rho[x + y * nx + zp * nx * ny];
        const rhoZm = rho[x + y * nx + zm * nx * ny];

        const dRhoDx = (rhoXp - rhoXm) / (xp === xm ? hx : 2 * hx);
        const dRhoDy = (rhoYp - rhoYm) / (yp === ym ? hy : 2 * hy);
        const dRhoDz = (rhoZp - rhoZm) / (zp === zm ? hz : 2 * hz);

        const dxx = (rhoXp - 2 * rhoC + rhoXm) / (xp === xm ? hx2 : hx2);
        const dyy = (rhoYp - 2 * rhoC + rhoYm) / (yp === ym ? hy2 : hy2);
        const dzz = (rhoZp - 2 * rhoC + rhoZm) / (zp === zm ? hz2 : hz2);

        const rhoXpYp = rho[xp + yp * nx + z * nx * ny];
        const rhoXpYm = rho[xp + ym * nx + z * nx * ny];
        const rhoXmYp = rho[xm + yp * nx + z * nx * ny];
        const rhoXmYm = rho[xm + ym * nx + z * nx * ny];

        const rhoXpZp = rho[xp + y * nx + zp * nx * ny];
        const rhoXpZm = rho[xp + y * nx + zm * nx * ny];
        const rhoXmZp = rho[xm + y * nx + zp * nx * ny];
        const rhoXmZm = rho[xm + y * nx + zm * nx * ny];

        const rhoYpZp = rho[x + yp * nx + zp * nx * ny];
        const rhoYpZm = rho[x + yp * nx + zm * nx * ny];
        const rhoYmZp = rho[x + ym * nx + zp * nx * ny];
        const rhoYmZm = rho[x + ym * nx + zm * nx * ny];

        const dxy = (rhoXpYp - rhoXpYm - rhoXmYp + rhoXmYm) /
          (xp === xm || yp === ym ? 4 * hx * hy : 4 * hx * hy);
        const dxz = (rhoXpZp - rhoXpZm - rhoXmZp + rhoXmZm) /
          (xp === xm || zp === zm ? 4 * hx * hz : 4 * hx * hz);
        const dyz = (rhoYpZp - rhoYpZm - rhoYmZp + rhoYmZm) /
          (yp === ym || zp === zm ? 4 * hy * hz : 4 * hy * hz);

        const gradNorm = Math.sqrt(dRhoDx * dRhoDx + dRhoDy * dRhoDy + dRhoDz * dRhoDz);
        const rhoSafe = Math.max(rhoC, EPS);
        const iriValue = gradNorm / Math.pow(rhoSafe, aValue);
        if (!Number.isFinite(iriValue)) {
          iri[idx] = 0;
          if (Number.isNaN(iriValue)) nanCount += 1;
          else infCount += 1;
        } else {
          iri[idx] = iriValue;
          if (iriValue < minIri) minIri = iriValue;
          if (iriValue > maxIri) maxIri = iriValue;
        }

        const lambda2 = eigenvalueLambda2(dxx, dyy, dzz, dxy, dxz, dyz);
        signLambda2Rho[idx] = Math.sign(lambda2) * rhoC;
      }
    }
  }

  if (minIri === Infinity) minIri = 0;
  if (maxIri === -Infinity) maxIri = 0;

  return { iri, signLambda2Rho, minIri, maxIri, nanCount, infCount };
}

function eigenvalueLambda2(
  dxx: number,
  dyy: number,
  dzz: number,
  dxy: number,
  dxz: number,
  dyz: number
): number {
  const p1 = dxy * dxy + dxz * dxz + dyz * dyz;
  if (p1 === 0) {
    const values = [dxx, dyy, dzz].sort((a, b) => a - b);
    return values[1];
  }
  const q = (dxx + dyy + dzz) / 3;
  const p2 =
    (dxx - q) * (dxx - q) +
    (dyy - q) * (dyy - q) +
    (dzz - q) * (dzz - q) +
    2 * p1;
  const p = Math.sqrt(p2 / 6);

  const b11 = (dxx - q) / p;
  const b22 = (dyy - q) / p;
  const b33 = (dzz - q) / p;
  const b12 = dxy / p;
  const b13 = dxz / p;
  const b23 = dyz / p;

  const detB =
    b11 * b22 * b33 +
    2 * b12 * b13 * b23 -
    b11 * b23 * b23 -
    b22 * b13 * b13 -
    b33 * b12 * b12;
  const r = detB / 2;

  let phi = 0;
  if (r <= -1) {
    phi = Math.PI / 3;
  } else if (r >= 1) {
    phi = 0;
  } else {
    phi = Math.acos(r) / 3;
  }

  const eig1 = q + 2 * p * Math.cos(phi);
  const eig3 = q + 2 * p * Math.cos(phi + (2 * Math.PI) / 3);
  const eig2 = 3 * q - eig1 - eig3;
  const values = [eig1, eig2, eig3].sort((a, b) => a - b);
  return values[1];
}

function buildIsosurface(
  iri: Float32Array,
  signLambda2Rho: Float32Array,
  dims: [number, number, number],
  bounds: { min: [number, number, number]; max: [number, number, number] },
  spacing: [number, number, number],
  isoValue: number,
  colorRange: { min: number; max: number },
  iriRange: { minIri: number; maxIri: number }
) {
  if (isoValue < iriRange.minIri || isoValue > iriRange.maxIri) {
    throw new Error(
      `No isosurface at isovalue ${isoValue}. IRI range is [${iriRange.minIri.toExponential(3)}, ${iriRange.maxIri.toExponential(3)}]`
    );
  }
  const [nx, ny, nz] = dims;
  const getIndex = (x: number, y: number, z: number) => x + y * nx + z * nx * ny;

  const mesh = isosurface.marchingCubes(dims, (x, y, z) => {
    return iri[getIndex(x, y, z)] - isoValue;
  });

  if (!mesh.positions?.length || !mesh.cells?.length) {
    throw new Error(
      `No surface generated for isovalue ${isoValue}. IRI range is [${iriRange.minIri.toExponential(3)}, ${iriRange.maxIri.toExponential(3)}]`
    );
  }

  const vertices = new Float32Array(mesh.positions.length * 3);
  const colors = new Float32Array(mesh.positions.length * 3);
  for (let i = 0; i < mesh.positions.length; i += 1) {
    const position = mesh.positions[i];
    if (!position || position.length < 3) {
      throw new Error("Invalid marching-cubes position data");
    }
    const [gx, gy, gz] = position;
    const worldX = bounds.min[0] + gx * spacing[0];
    const worldY = bounds.min[1] + gy * spacing[1];
    const worldZ = bounds.min[2] + gz * spacing[2];
    vertices[i * 3] = worldX;
    vertices[i * 3 + 1] = worldY;
    vertices[i * 3 + 2] = worldZ;

    const signValue = sampleTrilinear(signLambda2Rho, dims, gx, gy, gz);
    const color = mapSignColor(signValue, colorRange.min, colorRange.max);
    colors[i * 3] = color[0];
    colors[i * 3 + 1] = color[1];
    colors[i * 3 + 2] = color[2];
  }

  const indices = new Uint32Array(mesh.cells.length * 3);
  for (let i = 0; i < mesh.cells.length; i += 1) {
    indices[i * 3] = mesh.cells[i][0];
    indices[i * 3 + 1] = mesh.cells[i][1];
    indices[i * 3 + 2] = mesh.cells[i][2];
  }

  return { vertices, colors, indices };
}

function sampleTrilinear(
  field: Float32Array,
  dims: [number, number, number],
  x: number,
  y: number,
  z: number
): number {
  const [nx, ny, nz] = dims;
  const x0 = Math.floor(x);
  const y0 = Math.floor(y);
  const z0 = Math.floor(z);
  const x1 = Math.min(x0 + 1, nx - 1);
  const y1 = Math.min(y0 + 1, ny - 1);
  const z1 = Math.min(z0 + 1, nz - 1);
  const xd = x - x0;
  const yd = y - y0;
  const zd = z - z0;

  const idx = (ix: number, iy: number, iz: number) => ix + iy * nx + iz * nx * ny;
  const c000 = field[idx(x0, y0, z0)];
  const c100 = field[idx(x1, y0, z0)];
  const c010 = field[idx(x0, y1, z0)];
  const c110 = field[idx(x1, y1, z0)];
  const c001 = field[idx(x0, y0, z1)];
  const c101 = field[idx(x1, y0, z1)];
  const c011 = field[idx(x0, y1, z1)];
  const c111 = field[idx(x1, y1, z1)];

  const c00 = c000 * (1 - xd) + c100 * xd;
  const c01 = c001 * (1 - xd) + c101 * xd;
  const c10 = c010 * (1 - xd) + c110 * xd;
  const c11 = c011 * (1 - xd) + c111 * xd;
  const c0 = c00 * (1 - yd) + c10 * yd;
  const c1 = c01 * (1 - yd) + c11 * yd;
  return c0 * (1 - zd) + c1 * zd;
}

function mapSignColor(value: number, min: number, max: number): [number, number, number] {
  const clamped = Math.max(min, Math.min(max, value));
  const mid = 0;
  if (clamped <= mid) {
    const t = (clamped - min) / (mid - min || 1);
    return [0.1 + 0.2 * t, 0.6 + 0.3 * t, 0.9];
  }
  const t = (clamped - mid) / (max - mid || 1);
  return [0.9, 0.4 + 0.4 * (1 - t), 0.2 + 0.2 * (1 - t)];
}
