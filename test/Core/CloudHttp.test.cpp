#include <chrono>
#include <thread>
#include <ctime>
#include <string>
#include <sstream>
#include <iomanip>
#include <iostream>
#include "gtest/gtest.h"
#include "QPandaConfig.h"
#include "Core/QuantumCloud/QCloudMachine.h"

using namespace std;
USING_QPANDA

#if defined(USE_OPENSSL) && defined(USE_CURL)

#include <curl/curl.h>
#include <openssl/ec.h>
#include <openssl/ecdsa.h>
#include <openssl/evp.h>
#include <openssl/bio.h>
#include <openssl/pem.h>
#include <openssl/sha.h>
#include <openssl/err.h>
#include <openssl/objects.h>
#include <openssl/x509.h>
#include <openssl/crypto.h>

int hex_to_bin(const char* hex, unsigned char* bin, size_t len) {
    size_t i;
    for (i = 0; i < len; i++) {
        sscanf(hex + 2 * i, "%2hhx", &bin[i]);
    }

    return 1;
}

void openssl_test()
{
    const char* private_key_hex = "3041020100301306072a8648ce3d020106082a8648ce3d030107042730250201010420ecf1e6a65ac33acbda328d2269ea493d838d9f884cbe19d37b4c7db432d748d8";
    size_t der_len = strlen(private_key_hex) / 2;
    unsigned char* der_buf = (unsigned char*)malloc(der_len);
    hex_to_bin(private_key_hex, der_buf, der_len);

    EVP_PKEY* evp_private_key = d2i_PrivateKey(EVP_PKEY_EC, NULL, const_cast<const unsigned char**>(&der_buf), der_len);
    if (evp_private_key == NULL) {
        cout << "evp_private_key == NULL" << endl;
        unsigned long err = ERR_get_error();
        char err_str[256];
        ERR_error_string_n(err, err_str, sizeof(err_str));
        printf("Error decoding private key: %s\n", err_str);
        return;
    }

    EC_KEY* ec_key = EVP_PKEY_get1_EC_KEY(evp_private_key);
    if (ec_key == NULL)
    {
        cout << "ec_key == NULL" << endl;
        unsigned long err = ERR_get_error();
        char err_str[256];
        ERR_error_string_n(err, err_str, sizeof(err_str));
        printf("Error decoding private key: %s\n", err_str);
        return;
    }

    std::cout << ec_key << endl;
    std::cout << evp_private_key << endl;

    const BIGNUM *priv_key_bn = EC_KEY_get0_private_key(ec_key);
    const EC_POINT *pub_key_point = EC_KEY_get0_public_key(ec_key);

    char *priv_key_hex = BN_bn2hex(priv_key_bn);
    printf("priv_key_hex: %s\n", priv_key_hex);

    char *pub_key_hex = EC_POINT_point2hex(EC_KEY_get0_group(ec_key), pub_key_point, POINT_CONVERSION_UNCOMPRESSED, NULL);
    printf("pub_key_hex: %s\n", pub_key_hex);
    return;
}

TEST(CloudHttp, Cluster)
{
    //QCloudMachine machine;
    auto machine = QCloudMachine();
    machine.init("302e020100301006072a8648ce3d020106052b8104001c041730150201010410d0513887336bab9e1fdc4a2376746e8c/11028", true);
    
    machine.set_qcloud_api("https://qcloud4test.originqc.com");

    auto qlist = machine.allocateQubits(4);
    auto clist = machine.allocateCBits(4);
    auto measure_prog = QProg();
    measure_prog << HadamardQCircuit(qlist)
        << CZ(qlist[2], qlist[3])
        << Measure(qlist[0], clist[0]);

    auto pmeasure_prog = QProg();
    pmeasure_prog << HadamardQCircuit(qlist)
        << CZ(qlist[1], qlist[3])
        << RX(qlist[2], PI / 4)
        << RX(qlist[1], PI / 4);

    auto result0 = machine.full_amplitude_measure(measure_prog, 100);
    cout << "full_amplitude_measure result : "  << endl;
    for (auto val : result0)
        cout << val.first << " : " << val.second << endl;

    auto result1 = machine.full_amplitude_pmeasure(pmeasure_prog, { 0, 1, 2 });
    cout << "full_amplitude_pmeasure result : " << endl;
    for (auto val : result1)
        cout << val.first << " : " << val.second << endl;

    auto result2 = machine.partial_amplitude_pmeasure(pmeasure_prog, { "0", "1", "2" });
    cout << "partial_amplitude_pmeasure result : " << endl;
    for (auto val : result2)
        cout << val.first << " : " << val.second << endl;

    auto result3 = machine.single_amplitude_pmeasure(pmeasure_prog, "0");
    cout << "single_amplitude_pmeasure result : " << endl;
    cout << "0 : " << result3 << endl;

    machine.set_noise_model(NOISE_MODEL::BIT_PHASE_FLIP_OPRATOR, { 0.01 }, { 0.02 });
    auto result41 = machine.noise_measure(measure_prog, 100);
    cout << "noise_measure result : " << endl;
    for (auto val : result41)
        cout << val.first << " : " << val.second << endl;

    auto result4 = machine.real_chip_measure(measure_prog, 1000);
    cout << "real_chip_measure result : " << endl;
    for (auto val : result4)
        cout << val.first << " : " << val.second << endl;

    auto result5 = machine.get_state_tomography_density(measure_prog, 1000);
    cout << "get_state_tomography_density result : " << endl;
    for (auto val : result5)
    {
        for (auto val1 : val)
        {
            cout << val1 << endl;
        }
    }

    auto result6 = machine.get_state_fidelity(measure_prog, 1000);
    cout << "fidelity : " << result6 << endl;

    machine.finalize();
}

#endif // USE_CURL

