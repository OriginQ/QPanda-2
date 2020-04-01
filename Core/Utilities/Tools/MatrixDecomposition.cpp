#include <math.h>
#include <Eigen/KroneckerProduct>
#include "Core/Utilities/Tools/MatrixDecomposition.h"
USING_QPANDA

static void upper_partition(int order, MatrixOperator &entries)
{
    auto index = (int)std::log2(entries.size() + 1) - (int)std::log2(order) - 1;

    for (auto cdx = 0; cdx < order - 1; ++cdx)
    {
        for (auto rdx = 0; rdx < order - cdx - 1; ++rdx)
        {
            auto Entry = entries[cdx][rdx];

            Entry.first += order;
            Entry.second[index] = MatrixUnit::SINGLE_P1;
                
            entries[cdx + order].emplace_back(Entry);
        }
    }
}


static bool entry_requirement(const MatrixSequence& units, int udx, int cdx)
{
    int M = 0;
    while (cdx)
    {
        cdx >>= 1;
        M += cdx ? 1 : 0;
    }

    //if 1 ≤ j ≤ m and cj = lj' = 1 , return true
    auto unit = units[units.size() - udx - 1];
    return udx >= 1
        && udx <= M
        && cdx == 1 
        && unit == MatrixUnit::SINGLE_P1;
}

static bool steps_requirement(const MatrixSequence& units, int udx, int cdx)
{
    int M = 0;
    while (cdx)
    {
        cdx >>= 1;
        M += cdx ? 1 : 0;
    }

    //if j = n and none of cn...cm+1 is 1 , return true
    if (udx != units.size() - 1)
    {
        return false;
    }
    else
    {
        auto iter = std::find(units.begin(), units.end() - M, MatrixUnit::SINGLE_P1);
        return (units.end() - M) == iter;
    }
}

static void under_partition(int order, MatrixOperator& entries)
{
    auto qubits =(int)std::log2(entries.size() + 1);

    for (auto cdx = 1; cdx < order; ++cdx)
    {
        if (cdx & 1)
        {
            for (auto rdx = 0; rdx < order; ++rdx)
            {
                auto entry = entries[0][rdx + order - 1].first ^ cdx;
                entries[cdx].emplace_back(make_pair(entry, entries[cdx - 1][rdx + order - 1].second));
            }

            auto &units = entries[cdx].back().second;
            for (auto idx = 0; idx < (int)std::log2(order); ++idx)
            {
                if ((cdx >> idx) & 1)
                {
                    units[qubits - idx - 1] = MatrixUnit::SINGLE_P1;
                }
                else
                {
                    units[qubits - idx - 1] = MatrixUnit::SINGLE_I2;
                }
            }
        }
        else
        {
            for (auto rdx = 0; rdx < order; ++rdx)
            {
                auto units = entries[0][rdx + order].second;
                auto entry = entries[0][rdx + order].first ^ cdx;

                for (auto udx = 0; udx < qubits; ++udx)
                {
                    bool steps_accord = entry_requirement(units, udx, cdx);
                    bool entry_accord = entry_requirement(units, udx, cdx);

                    if (steps_accord)
                    {
                        units[udx] = MatrixUnit::SINGLE_P1;
                    }
                    else if (entry_accord)
                    {
                        units[udx] = MatrixUnit::SINGLE_P0;
                    }
                    else
                    {
                    }
                }

                entries[cdx].emplace_back(make_pair(entry, units));
            }

            auto &units = entries[0].back().second;
            for (auto idx = 0; idx < (int)std::log2(order); ++idx)
            {
                if ((cdx >> idx) & 1)
                {
                    units[qubits - idx - 1] = MatrixUnit::SINGLE_P1;
                }
                else
                {
                    units[qubits - idx - 1] = MatrixUnit::SINGLE_I2;
                }
            }
        }
    }
}

static void voluation(Eigen::MatrixXcf& matrix, MatrixOperator& entries)
{
    auto qubits = (int)std::log2(matrix.rows());

    MatrixSequence Cns(qubits, MatrixUnit::SINGLE_I2);
    Cns.back() = MatrixUnit::SINGLE_V2;
    entries.front().emplace_back(make_pair(1, Cns));

    ColumnOperator& column = entries.front();
    for (auto idx = 1; idx < qubits; ++idx)
    {
        size_t path = 1ull << idx;
        for (auto opt = 0; opt < (1 << idx) - 1; ++opt)
        {
            auto units = column[opt].second;

            // 1 : none of cn−1, . . . , c1 equals 1
            // * : otherwise
            auto iter = std::find(units.begin() + 1, units.end(), MatrixUnit::SINGLE_P1);
            if (units.end() == iter)
            {
                units.front() = MatrixUnit::SINGLE_P1;;
            }
            else
            {
                units.front() = MatrixUnit::SINGLE_I2;;
            }

            column.emplace_back(make_pair(opt + path + 1, units));
        }
          
        MatrixSequence Lns(qubits, MatrixUnit::SINGLE_I2);
        Lns[qubits - idx - 1] = MatrixUnit::SINGLE_V2;

        column.emplace_back(make_pair((1ull << idx), Lns));
    } 
}

static void controller(MatrixSequence &sequence, const Eigen::Matrix2cf U2, Eigen::MatrixXcf &matrix)
{
    Eigen::Matrix2cf P0;
    Eigen::Matrix2cf P1;
    Eigen::Matrix2cf I2;

    P0 << Eigen::scomplex(1, 0), Eigen::scomplex(0, 0), 
          Eigen::scomplex(0, 0), Eigen::scomplex(0, 0);
    P1 << Eigen::scomplex(0, 0), Eigen::scomplex(0, 0), 
          Eigen::scomplex(0, 0), Eigen::scomplex(1, 0);
    I2 << Eigen::scomplex(1, 0), Eigen::scomplex(0, 0), 
          Eigen::scomplex(0, 0), Eigen::scomplex(1, 0);

    std::map<MatrixUnit, std::function<Eigen::Matrix2cf()>> mapping =
    {
        { MatrixUnit::SINGLE_P0, [&]() {return P0; } },
        { MatrixUnit::SINGLE_P1, [&]() {return P1; } },
        { MatrixUnit::SINGLE_I2, [&]() {return I2; } },
        { MatrixUnit::SINGLE_V2, [&]() {return U2 - I2; } }
    };

    auto order = sequence.size();
    Eigen::MatrixXcf Un = Eigen::MatrixXcf::Identity(1, 1);
    Eigen::MatrixXcf In = Eigen::MatrixXcf::Identity(1ull << order, 1ull << order);

    for (const auto &val : sequence)
    {
        Eigen::Matrix2cf M2 = mapping.find(val)->second();
        Un = Eigen::kroneckerProduct(Un, M2).eval();
    }

    matrix = In + Un;
}

static void operation(Eigen::MatrixXcf& matrix, MatrixOperator& entries)
{
    for (auto cdx = 0; cdx < entries.size(); ++cdx)
    {
        for (auto idx = 0; idx < entries[cdx].size(); ++idx)
        {
            auto rdx = entries[cdx][idx].first;
            auto opt = entries[cdx][idx].second;

            if (Eigen::scomplex(0, 0) == matrix(rdx, cdx))
            {
                continue;;
            }
            else
            {
                auto order = opt.size();

                Eigen::Matrix2cf C2; /*placeholder*/
                C2 << Eigen::scomplex(0, 1), Eigen::scomplex(0, 1), 
                      Eigen::scomplex(0, 1), Eigen::scomplex(0, 1); 

                Eigen::MatrixXcf Cn;
                controller(opt, C2, Cn);

                Qnum indices(2);
                for (Eigen::Index index = 0; index < (1ull << order); ++index)
                {
                    if (Cn(rdx, index) != Eigen::scomplex(0, 0))
                    {
                        indices[index == rdx] = index;
                    }
                }

                Eigen::scomplex C0 = matrix(indices[0], cdx);
                Eigen::scomplex C1 = matrix(indices[1], cdx);

                Eigen::scomplex V11 = std::conj(C0) / std::sqrt(std::norm(C0) + std::norm(C1));
                Eigen::scomplex V12 = std::conj(C1) / std::sqrt(std::norm(C0) + std::norm(C1));
                Eigen::scomplex V21 =  C1 / std::sqrt(std::norm(C0) + std::norm(C1));
                Eigen::scomplex V22 = -C0 / std::sqrt(std::norm(C0) + std::norm(C1));

                Eigen::Matrix2cf V2;
                V2 << V11 , V12 , V21 , V22;

                Eigen::MatrixXcf Un;
                controller(opt, V2, Un);

                matrix = Un * matrix;
            }
        }
    }
}

static void partition(Eigen::MatrixXcf& sub_matrix, MatrixOperator &entries)
{
    Eigen::Index order = sub_matrix.rows();
    if (1 == order)
    {
        return;
    }
    else
    {
        Eigen::MatrixXcf corner = sub_matrix.topLeftCorner(order / 2, order / 2);

        partition(corner, entries);

        upper_partition(order / 2, entries);
        under_partition(order / 2, entries);
    }
}

static void general_scheme(Eigen::MatrixXcf& matrix)
{
    MatrixOperator entries;
    for (auto idx = 1; idx < matrix.cols(); ++idx)
    {
        ColumnOperator Co;
        entries.emplace_back(Co);
    }

    voluation(matrix, entries);
    partition(matrix, entries);
    operation(matrix, entries);
}

void QMatrix::decompose()
{
    auto order = (int)std::log2(this->size());

    Eigen::MatrixXcf matrix = Eigen::MatrixXcf::Zero(order, order);
    for (auto rdx = 0; rdx < order; ++rdx)
    {
        for (auto cdx = 0; cdx < order; ++cdx)
        {
            matrix(rdx, cdx) = this->at(rdx*order + cdx);
        }
    }

    if (!matrix.isUnitary(1e-3))
    {
        QCERR("Non-unitary matrix");
        return;
    }

    general_scheme(matrix);
}