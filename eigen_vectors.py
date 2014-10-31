import numpy as np

def main():
  # f = np.array([[1, 2], [3, 4]]);
  f = np.array([[0.544200, 0.086533, -0.038167],
                [0.096000, -0.149267, 0.466500],
                [-0.015000, -0.322667, -10.193000]]);
  cg = np.dot(f.T, f);
  eigen_values, eigen_vectors = np.linalg.eig(cg);

  print eigen_vectors

if __name__ == '__main__':
  main()
