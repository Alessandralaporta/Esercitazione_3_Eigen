#include <iostream>
#include <Eigen/Dense>
using namespace Eigen;
using namespace std;

Vector2d solve_palu(const Matrix2d& A, const Vector2d& b) {
    PartialPivLU<Matrix2d> lu(A);
    return lu.solve(b);
}

Vector2d solve_qr(const Matrix2d& A, const Vector2d& b) {
    HouseholderQR<Matrix2d> qr(A);
    return qr.solve(b);
}

double relative_error(const Vector2d& x_computed, const Vector2d& x_true) {
    return (x_computed - x_true).norm() / x_true.norm();
}

void solveBoth(const Matrix2d A, const Vector2d b, const Vector2d sol, const string& name) {
    Vector2d xPALU = solve_palu(A, b);
    double errPALU = relative_error(xPALU, sol);

    Vector2d xQR = solve_qr(A, b);
    double errQR = relative_error(xQR, sol);

    std::cout << "Solution for " << name << " with PALU decomposition is:\n" << xPALU << "\n";
    std::cout << "Relative error for " << name << " with PALU decomposition is: " << errPALU << "\n\n";
    std::cout << "Solution for " << name << " with QR decomposition is:\n" << xQR << "\n";
    std::cout << "Relative error for " << name << " with QR decomposition is: " << errQR << "\n\n";

}

int main()
{
    Vector2d sol;
    sol  << -1.0e+0, -1.0e+00;

    Matrix2d A1;
    Vector2d b1;
    A1 << 5.547001962252291e-01, -3.770900990025203e-02,   8.320502943378437e-01, -9.992887623566787e-01;
    b1 << -5.169911863249772e-01,   1.672384680188350e-01;

    Matrix2d A2;
    Vector2d b2;
    A2 << 5.547001962252291e-01, -5.540607316466765e-01, 8.320502943378437e-01, -8.324762492991313e-01;
    b2 << -6.394645785530173e-04, 4.259549612877223e-04;

    Matrix2d A3;
    Vector2d b3;
    A3 << 5.547001962252291e-01, -5.547001955851905e-01, 8.320502943378437e-01, -8.320502947645361e-01;
    b3 << -6.400391328043042e-10, 4.266924591433963e-10;

    solveBoth(A1, b1, sol, "System 1");
    solveBoth(A2, b2, sol, "System 2");
    solveBoth(A3, b3, sol, "System 3");

    return 0;
}

