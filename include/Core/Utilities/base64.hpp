/*
 * Simple base64 encoder/decoder in C++11
 * copyright(c) 2017 Hajime UCHIMURA (@nikq, nikutama@gmail.com)
 *
 * MIT License
 */

#ifndef _BASE64_H_
#define _BASE64_H_

#include <time.h>
#include <stdio.h>
#include <vector>
#include <algorithm>
#include <cassert>
#include <inttypes.h>

namespace Base64{

    uint8_t uc_decode (uint8_t uc)
    {
        if ('A' <= uc && uc <= 'Z')
        {
            return uc - 'A' ;
        }
        if ('a' <= uc && uc <= 'z')
        {
            return 26 + uc - 'a' ;
        }
        if ('0' <= uc && uc <= '9')
        {
            return 52 + uc - '0' ;
        }
        if (uc == '+')
        {
            return 62 ;
        }
        if (uc == '/')
        {
            return 63 ;
        }
        assert (false) ;
        return 0 ;
    }

    uint8_t uc_encode (uint8_t uc)
    {
        if (uc < 26)
        {
            return 'A' + uc ;
        }
        if (uc < 52)
        {
            return 'a' + uc - 26 ;
        }
        if (uc < 62)
        {
            return '0' + uc - 52 ;
        }
        if (uc == 62)
        {
            return '+' ;
        }
        if (uc == 63)
        {
            return '/' ;
        }
        assert (false) ;
        return '=' ;
    }

    /**
         * Encodes supplied bytes into base64 encoded octets.
         */
    std::vector<uint8_t> encode(const void *input, size_t length)
    {
        const uint8_t * bin = static_cast<const uint8_t *> (input) ;
        if (length == 0)
        {
            return std::vector<uint8_t>();
        }

        size_t cntFullBlocks = length / 3 ;
        std::vector<uint8_t>    result (4 * cntFullBlocks) ;

        for (size_t i = 0 ; i < cntFullBlocks ; ++i)
        {
            auto b0 = bin [3 * i + 0] ;
            const uint8_t a012345 = (b0 >> 2) & 0x3Fu ;
            const uint8_t a67____ = (b0 << 4) & 0x3Fu ;
            auto b1 = bin [3 * i + 1] ;
            const uint8_t b__0123 = (b1 >> 4) & 0x3Fu ;
            const uint8_t b4567__ = (b1 << 2) & 0x3Fu ;
            auto b2 = bin [3 * i + 2] ;
            const uint8_t c____01 = (b2 >> 6) & 0x3Fu ;
            const uint8_t c234567 = (b2 >> 0) & 0x3Fu ;

            result [4 * i + 0] = uc_encode (a012345) ;
            result [4 * i + 1] = uc_encode (a67____ | b__0123) ;
            result [4 * i + 2] = uc_encode (b4567__ | c____01) ;
            result [4 * i + 3] = uc_encode (c234567) ;
        }
        auto lastIdx = 3 * cntFullBlocks ;
        assert (lastIdx <= length) ;
        switch (length - lastIdx)
        {
        case 0:
            // NO-OP
            break ;
        case 1:
        {
            auto b0 = bin [lastIdx + 0] ;
            const uint8_t a012345 = (b0 >> 2) & 0x3Fu ;
            const uint8_t a67____ = (b0 << 4) & 0x3Fu ;
            result.emplace_back (uc_encode (a012345)) ;
            result.emplace_back (uc_encode (a67____)) ;
            result.emplace_back ('=') ;
            result.emplace_back ('=') ;
            break ;
        }
        case 2:
        {
            auto b0 = bin [lastIdx + 0] ;
            const uint8_t a012345 = (b0 >> 2) & 0x3Fu ;
            const uint8_t a67____ = (b0 << 4) & 0x3Fu ;
            auto b1 = bin [lastIdx + 1] ;
            const uint8_t b__0123 = (b1 >> 4) & 0x3Fu ;
            const uint8_t b4567__ = (b1 << 2) & 0x3Fu ;
            result.emplace_back (uc_encode (a012345)) ;
            result.emplace_back (uc_encode (a67____ | b__0123)) ;
            result.emplace_back (uc_encode (b4567__)) ;
            result.emplace_back ('=') ;
            break ;
        }
        default:
            assert (false) ;
        }
        return result ;
    }

    /**
        * Encodes supplied bytes into base64 encoded octets.
        */
    inline std::vector<uint8_t> encode (const std::vector<uint8_t> &bin)
    {
        return encode (bin.data (), bin.size ()) ;
    }


    /**
         * Decodes supplied base64 encoded octets into raw bytes.
         */
    std::vector<uint8_t> decode(const void *input, size_t length)
    {
        const uint8_t * b64 = static_cast<const uint8_t *> (input) ;
        //uint8_t * b64 = (uint8_t *)input ;
        if (length == 0)
            return std::vector<uint8_t>();
        assert (length % 4 == 0) ;
        assert (4 <= length) ;
        size_t cntFullBlock = length / 4 - 1 ;

        std::vector<uint8_t>    result (3 * cntFullBlock) ;
        for (size_t i = 0 ; i < cntFullBlock ; ++i)
        {
            const uint8_t a012345 = uc_decode (b64 [4 * i + 0]);
            const uint8_t a670123 = uc_decode (b64 [4 * i + 1]);
            const uint8_t b456701 = uc_decode (b64 [4 * i + 2]);
            const uint8_t c234567 = uc_decode (b64 [4 * i + 3]);

            result [3 * i + 0] = (a012345 << 2) | (a670123 >> 4) ;
            result [3 * i + 1] = (a670123 << 4) | (b456701 >> 2) ;
            result [3 * i + 2] = (b456701 << 6) | (c234567 >> 0) ;
        }
        // Process the last input block.
        auto lastIdx = 4 * cntFullBlock ;
        assert (lastIdx < length) ;
        if (b64 [lastIdx + 3] == '=') {
            if (b64 [lastIdx + 2] == '=') {
                // Last block contains 1 octet.
                const uint8_t a012345 = uc_decode (b64 [lastIdx + 0]);
                const uint8_t a670123 = uc_decode (b64 [lastIdx + 1]);
                result.emplace_back ((a012345 << 2) | (a670123 >> 4)) ;
            }
            else {
                // Last block contains 2 octet.
                const uint8_t a012345 = uc_decode (b64 [lastIdx + 0]);
                const uint8_t a670123 = uc_decode (b64 [lastIdx + 1]);
                const uint8_t b456701 = uc_decode (b64 [lastIdx + 2]);
                result.emplace_back ((a012345 << 2) | (a670123 >> 4)) ;
                result.emplace_back ((a670123 << 4) | (b456701 >> 2)) ;
            }
        }
        else {
            // Full block (contains 3 octet).
            const uint8_t a012345 = uc_decode (b64 [lastIdx + 0]);
            const uint8_t a670123 = uc_decode (b64 [lastIdx + 1]);
            const uint8_t b456701 = uc_decode (b64 [lastIdx + 2]);
            const uint8_t c234567 = uc_decode (b64 [lastIdx + 3]);

            result.emplace_back ((a012345 << 2) | (a670123 >> 4)) ;
            result.emplace_back ((a670123 << 4) | (b456701 >> 2)) ;
            result.emplace_back ((b456701 << 6) | (c234567 >> 0)) ;
        }
        return result ;
    }

    /**
        * Decodes supplied base64 encoded octets into raw bytes.
        */
    inline std::vector<uint8_t> decode (const std::vector<uint8_t> &input)
    {
        return decode (input.data (), input.size ()) ;
    }
}




#endif  /* _BASE64_H_ */
