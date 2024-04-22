#define _USE_MATH_DEFINES

#include <iostream>
#include <cmath>
#include <vector>
#include <Eigen/Dense>
#include <complex>
#include <algorithm>

const double a = 0;
const double b = M_PI / 2;
const double EPS = 1e-7;
const int N = 5;

double I()
{
    return 9. /  26 * ( 1 + 2./3 * pow(M_E, -M_PI / 2));
}

double f(double x)
{
    return pow(M_E, -x) * pow(cos(x), 5);
}

double middleRect(int n, double h)
{
    double res = 0;
    for (int i = 0; i <= n - 1; i++)
    {
        double x = a + (double)i * h + h / 2;
        res += f(x);
    }

    return res * h;
}

double simpson(int n, double h)
{
    // n % 2 == 0 every time
    double summOdd = 0, summEven = 0;
    for (int i = 1; i <= n - 1; i++)
    {
        double x = a + (double)i * h;
        summOdd += (i % 2 == 1) * f(x);
        summEven += (i % 2 == 0) * f(x);
    }

    return h / 3 * (f(a) + f(b) + 4. * summOdd + 2. * summEven);
}

Eigen::VectorXd getLegendreKoeffs()
{
    Eigen::MatrixXd A(N, N);
    Eigen::VectorXd b(N);

    for (int startC = 0; startC < N; startC++)
    {
        for (int i = N + startC; i >= startC; i--)
        {
            if (i != startC + N)
            {
                A(startC, N + startC - i - 1) = 1. * (i % 2 == 0) * (1. / (i + 1));
            }
            else
            {
                b(startC) = -1. * (i % 2 == 0) * (1. / (i + 1));
            }
        }
    }

    Eigen::VectorXd x = A.colPivHouseholderQr().solve(b);
    return x;
}

std::vector<double> SolveLegendre()
{
    Eigen::VectorXd coeffs = getLegendreKoeffs();
    Eigen::MatrixXd companionMatrix(N, N);
    for (int i = 0; i < N; ++i)
    {
        for (int j = 0; j < N; ++j)
        {
            companionMatrix(i, j) = 0;
        }
    }

    for (int i = 1; i < N; ++i)
    {
        companionMatrix(i, i - 1) = 1;
    }
    
    for (int i = 0; i < N; ++i)
    {
        companionMatrix(i, N - 1) = -coeffs[N - i - 1];
    }

    Eigen::EigenSolver<Eigen::MatrixXd> es(companionMatrix);
    Eigen::VectorXcd roots = es.eigenvalues();

    std::vector<double>res(N);
    for (size_t i = 0; i < roots.size(); ++i)
    {
        res[i] = roots[i].real();
    }

    return res;
}

std::vector<double> getWeights(std::vector<double>& nodes)
{
    std::vector<double> weights(N);
    for (int i = 0; i < N; i++)
    {
        std::vector<double> nodesW;
        double del = 1;
        for (int j = 0; j < N; j++)
        {
            if (j == i)continue;
            nodesW.push_back(nodes[j]);
            del *= (nodes[i] - nodes[j]);
        }

        for (int num0 = 0; num0 < N; num0+=2)
        {
            double res = 0;
            std::vector<int>bitmask(nodesW.size(), 1);
            for (int j = 0; j < num0; j++)
            {
                bitmask[j] = 0;
            }

            do
            {
                double umn = 1;
                for (int j = 0; j < bitmask.size(); j++)
                {
                    if (bitmask[j])
                        umn *= nodesW[j];
                }

                res += umn;
            } while (std::next_permutation(bitmask.begin(), bitmask.end()));

            weights[i] += (2. / (num0 + 1)) * res;
        }

        weights[i] = weights[i] / del;
        if (N % 2 == 0)weights[i] *= -1;
    }

    return weights;
}

int main()
{
    std::vector<std::pair<double(*) (int, double), int >> kfs = { {middleRect, 2}, {simpson, 4} };
    for (auto values : kfs)
    {
        int m = values.second;
        std::cout << ((m == 2) ? "Middle rectangles method\n" : "Simpson's method\n");
        double h = (b - a);
        int n = 1;
        double r = NULL;
        do
        {
            n *= 2;
            h /= 2;
            double qh = values.first(n, h);
            if (r == NULL)
                std::cout << n << ' ' << h << ' ' << qh << ' ' << " - " << '\n';
            else
                std::cout << n << ' ' << h << ' ' << qh << ' ' << r << '\n';

            //std::cout << n << '\n';
            double qh2 = values.first(n * 2, h / 2);
            r = abs((qh2 - qh) / (pow(2, m) - 1));
        } while (abs(r) > EPS);

        //std::cout << values.first(n * 2, h / 2) << '\n';
        std::cout << abs(I() - values.first(n * 2, h / 2)) << '\n';
    }

    std::vector<double> nodes = SolveLegendre();
    std::vector<double> weights = getWeights(nodes);

    std::cout << "nodes: ";
    for (auto& i : nodes)
    {
        i = i * (b - a) / 2 + (a + b) / 2;
        std::cout << i << ' ';
    }
    std::cout << '\n';
    std::cout << "weights: ";
    for (auto& i : weights)
    {
        i *= (b - a) / 2;
        std::cout << i << ' ';
    }

    std::cout << '\n';

    double integral = 0;
    for (int i = 0; i < N; i++)
    {
        integral += f(nodes[i]) * weights[i];
    }


    std::cout << "correct integral value: " << I() << '\n';
    std::cout << "NAST value: " << integral << '\n';
    std::cout << "measurement: " << abs(I() - integral) << '\n';

}