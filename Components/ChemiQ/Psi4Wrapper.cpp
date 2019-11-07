#include "Components/ChemiQ/Psi4Wrapper.h"
#include "Core/Utilities/Tools/QString.h"

#include <math.h>

#if defined(_DEBUG)
#undef _DEBUG
#include <Python.h>
#define _DEBUG
#else
#include <Python.h>
#endif
#include <codecvt>

namespace QPanda {

Psi4Wrapper::Psi4Wrapper()
{

}

void Psi4Wrapper::initialize(const std::string &chemiq_dir)
{
#ifdef _MSC_VER
    if (!chemiq_dir.empty())
    {
        using convert_typeX = std::codecvt_utf8<wchar_t>;
        std::wstring_convert<convert_typeX, wchar_t> converterX;

        auto w_chemiq_dir = converterX.from_bytes(chemiq_dir);
        Py_SetPath(w_chemiq_dir.c_str());
    }
#endif
    
    Py_Initialize();

    /* Alter sys path eventment */
    std::string chdir_cmd = std::string("sys.path.append(\"")
        + chemiq_dir + "\")";
    PyRun_SimpleString("import sys");
    PyRun_SimpleString(chdir_cmd.c_str());
    //PyRun_SimpleString("print(sys.path)");
}

bool Psi4Wrapper::run()
{
    auto u_name = PyUnicode_FromString("psi4_wrapper");
    auto module = PyImport_Import(u_name);

    if (module == NULL)
    {
        PyErr_Print();
        m_last_error = "PyImport_Import() return NULL!";
        return false;
    }

    Py_DECREF(u_name);
    auto call_func = PyObject_GetAttrString(module, "run_psi4");
    Py_DECREF(module);
    if (call_func == NULL)
    {
        PyErr_Print();
        m_last_error = "PyObject_GetAttrString() return NULL!";
        return false;
    }

    auto args = Py_BuildValue(
        "({s:s,s:i,s:i,s:s,s:d})",
        "mol",m_molecule.c_str(),
        "multiplicity", m_multiplicity,
        "charge", m_charge,
        "basis", m_basis.c_str(),
        "EQ_TOLERANCE", m_eq_tolerance);
    auto result = PyObject_Call(call_func, args, NULL);

    if (result == NULL)
    {
        PyErr_Print();
        m_last_error = "PyObject_Call() return NULL!";
        return false;
    }

    int success = 1;
    char *value;

    PyArg_ParseTuple(result, "is", &success, &value);

    if (success == 0)
    {
        m_data = std::string(value);
    }
    else
    {
        m_last_error = std::string(value);
    }

    Py_DECREF(args);
    Py_DECREF(call_func);
    Py_DECREF(result);
    return success == 0;
}

void Psi4Wrapper::finalize()
{
    Py_Finalize();
}

}
