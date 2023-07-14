#include <vector>
#include <iomanip>
#include "Core/QuantumCloud/QCloudLog.h"
#include "Core/QuantumCloud/Signature.h"

using namespace std;

#if defined(USE_OPENSSL) && defined(USE_CURL)

#include <openssl/ec.h>
#include <openssl/ecdsa.h>
#include <openssl/evp.h>
#include <openssl/bio.h>
#include <openssl/pem.h>
#include <openssl/sha.h>
#include <openssl/err.h>
#include <openssl/x509.h>
#include <openssl/crypto.h>
#include <openssl/objects.h>

//convert the timestamp to a string in the format : YYYY-MM-DD
static string timestamp_to_string(time_t timestamp)
{
    tm* t = gmtime(&timestamp);
    stringstream ss;
    ss << t->tm_year + 1900 << '-' << setw(2) << setfill('0') << t->tm_mon + 1 << '-' << setw(2) << setfill('0') << t->tm_mday;
    return ss.str();
}

//get current timestamp
static time_t get_timestamp()
{
    time_t t = time(nullptr);
    return t;
}

//generate a random number with a specified number of digits
static std::string generate_random_number(int num_digits)
{
    srand(time(NULL));
    std::ostringstream os;
    for (int i = 0; i < num_digits; i++) {
        os << rand() % 10;
    }
    return os.str();
}

//SHA256 hash
std::unique_ptr<uint8_t[]> hash_message(const char* message, size_t message_len)
{
    std::unique_ptr<uint8_t[]> sha256_digest(new uint8_t[SHA256_DIGEST_LENGTH]);
    SHA256((const unsigned char*)message, message_len, sha256_digest.get());
    return sha256_digest;
}

//ECDSA signing the message
std::unique_ptr<uint8_t[]> sign_message(const uint8_t* sha256_digest, const EC_KEY* key_pair_obj, uint32_t& sign_length)
{
    std::unique_ptr<uint8_t[]> signature(new uint8_t[sign_length]);
    ECDSA_sign(0, sha256_digest, SHA256_DIGEST_LENGTH, signature.get(), &sign_length, (EC_KEY*)key_pair_obj);
    return signature;
}

//convert the hex data to a bytes array
std::unique_ptr<unsigned char[]> hex_to_bin(const char* hex, size_t len)
{
    std::unique_ptr<unsigned char[]> bin(new unsigned char[len]);

    for (size_t i = 0; i < len; i++)
        sscanf(hex + 2 * i, "%2hhx", &bin[i]);

    return bin;
}

static std::string sha256_with_ecdsa(const char *message, const char *private_key_hex)
{
    //OpenSSL_add_all_algorithms();

    size_t der_len = strlen(private_key_hex) / 2;
    auto der_buf = hex_to_bin(private_key_hex, der_len);
    auto der_buf_ptr = der_buf.get();

    EVP_PKEY* evp_private_key = d2i_PrivateKey(EVP_PKEY_EC, NULL, const_cast<const unsigned char**>(&der_buf_ptr), der_len);
    if (evp_private_key == NULL) 
    {
        unsigned long err = ERR_get_error();
        char err_str[256];
        ERR_error_string_n(err, err_str, sizeof(err_str));

        std::string error_msg = "Error decoding private key in d2i_PrivateKey: " + std::string(err_str);
        throw std::runtime_error(error_msg);
    }

    EC_KEY* ec_key = EVP_PKEY_get1_EC_KEY(evp_private_key);
    if (ec_key == NULL)
    {
        unsigned long err = ERR_get_error();
        char err_str[256];
        ERR_error_string_n(err, err_str, sizeof(err_str));

        std::string error_msg = "Error decoding private key in EVP_PKEY_get1_EC_KEY: " + std::string(err_str);
        throw std::runtime_error(error_msg);
    }

    const BIGNUM *priv_key_bn = EC_KEY_get0_private_key(ec_key);
    const EC_POINT *pub_key_point = EC_KEY_get0_public_key(ec_key);

    char *priv_key_hex = BN_bn2hex(priv_key_bn);
    char *pub_key_hex = EC_POINT_point2hex(EC_KEY_get0_group(ec_key), pub_key_point, POINT_CONVERSION_UNCOMPRESSED, NULL);

    //the signature size depends on the key
    uint32_t sign_length = ECDSA_size(ec_key);

    auto sha256_digest = hash_message(message, strlen(message));

    auto signature = sign_message(sha256_digest.get(), ec_key, sign_length);

    std::stringstream signature_stream;
    for (uint32_t i = 0; i < sign_length; i++)
        signature_stream << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(signature[i]);

    //verify the signature
    auto verification = ECDSA_verify(0, sha256_digest.get(), SHA256_DIGEST_LENGTH, signature.get(), sign_length, ec_key);
    if (verification != 1)
        throw std::runtime_error("signature verification failed");

    OPENSSL_free(pub_key_hex);
    OPENSSL_free(priv_key_hex);
    EVP_PKEY_free(evp_private_key);
    EC_KEY_free(ec_key);

    //EVP_cleanup();
    return signature_stream.str();
}

std::string qcloud_signature(const std::string& apikey)
{
    //separate private_key and user id
    string private_key = apikey.substr(0, apikey.find('/'));
    string user_id = apikey.substr(apikey.find('/') + 1);

    //generate a random number
    string random_str = generate_random_number(6);

    //get current timestamp
    time_t timestamp = get_timestamp();

    //convert the timestamp to a string in the format : YYYY-MM-DD
    string date_str = timestamp_to_string(timestamp);

    //construct the string to be encrypted
    string message = "rm=" + random_str + "&seid=" + user_id + "&tms=" + to_string(timestamp);

    //sha256 with ECDSA signature
    string signature = sha256_with_ecdsa(message.c_str(), private_key.c_str());

    //construct the final attribute curl header value
    string credential_str = user_id + "/" + date_str + "/" + to_string(timestamp) + "/" + random_str;
    string oqc_signature_str = "Authorization:Credential=" + credential_str + ",OqcSignature=" + signature;

    //cout << oqc_signature_str << endl;
    return oqc_signature_str;
}

#endif
