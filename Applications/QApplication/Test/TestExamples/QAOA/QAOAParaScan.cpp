#include "QAOAParaScan.h"
#include "Applications/QApplication/QAOA/QAOA.h"
#include "QString.h"
#include "RJson/RJson.h"
#include "mpi.h"
#include "tag_marco.h"

namespace QPanda
{

    QAOAParaScan::QAOAParaScan()
    {

    }

    bool QAOAParaScan::exec(QAOA &qaoa)
    {
        QScanPara p;
        p.two_para = get2P(m_para[STR_2P]);
        vector_i pos = get2Pos(m_para[STR_2POS]);
        p.pos1 = static_cast<size_t>(pos[0]);
        p.pos2 = static_cast<size_t>(pos[1]);
        p.filename = m_output_file;

        if (m_para.HasMember(STR_KEYS))
        {
            p.keys = getKeys(m_para[STR_KEYS]);
        }

        if (m_use_mpi)
        {
            MPI_Init(nullptr, nullptr);
            int size;
            int rank;
            MPI_Comm_size(MPI_COMM_WORLD, &size);
            MPI_Comm_rank(MPI_COMM_WORLD, &rank);

            if (size > 1)
            {
                p.filename = std::to_string(rank) + p.filename;
            }
            auto delta = fabs(p.two_para.x_max - p.two_para.x_min) / size;
            if (rank != 0)
            {
                p.two_para.x_min = p.two_para.x_min + rank * delta + p.two_para.x_step;
            }
            else
            {
                p.two_para.x_min = p.two_para.x_min + rank * delta;
            }

            p.two_para.x_max = p.two_para.x_min + delta;

        }

        qaoa.scan2Para(p);

        if (m_use_mpi)
        {
            MPI_Finalize();
        }

        return true;
    }

    QTwoPara QAOAParaScan::get2P(rapidjson::Value &value)
    {
        QTwoPara tp;
        tp.x_min  = value[STR_X_MIN].GetDouble();
        tp.x_max  = value[STR_X_MAX].GetDouble();
        tp.x_step = value[STR_X_STEP].GetDouble();
        tp.y_min  = value[STR_Y_MIN].GetDouble();
        tp.y_max  = value[STR_Y_MAX].GetDouble();
        tp.y_step = value[STR_Y_STEP].GetDouble();

        return tp;
    }

    vector_i QAOAParaScan::get2Pos(rapidjson::Value &value)
    {
        vector_i vec;
        vec.push_back(value[0].GetInt());
        vec.push_back(value[1].GetInt());

        return vec;
    }

    vector_i QAOAParaScan::getKeys(rapidjson::Value &value)
    {
        vector_i vec;
        for (rapidjson::SizeType i = 0; i < value.Size(); i++)
        {
            std::string key;
            RJson::GetStr(key, i, &value);
            QString item(key);
            bool ok = false;
            int value = item.toInt(&ok, QString::BIN);
            if (ok)
            {
                vec.push_back(value);
            }
        }

        return vec;
    }

}
