#pragma once
#include "string"
#include <codecvt>
#include <locale>

#ifdef _MSC_VER
#include <windows.h>
#endif

QPANDA_BEGIN

/**
* @brief convert unicode string to UTF8 string
* @param[in] std::wstring& the source wstring encoded by Unicode
* @return string Converted String
*/
inline std::string UnicodeToUTF8(const std::wstring & wstr)
{
	std::string ret;
	try {
		std::wstring_convert< std::codecvt_utf8<wchar_t> > wcv;
		ret = wcv.to_bytes(wstr);
	}
	catch (const std::exception & e) {
		std::cerr << e.what() << std::endl;
	}
	return ret;
}

/**
* @brief convert UTF8 string to wide string
* @param[in] std::string& the source string
* @return wstring Converted wide String
*/
inline std::wstring utf8ToWstring(const std::string& str)
{
	std::wstring_convert<std::codecvt_utf8<wchar_t>> myconv;
	return myconv.from_bytes(str);
}

#ifdef _MSC_VER
/**
* @brief convert UTF8 string to gbk
* @param[in] char* the source string
* @return std::string Converted string
*/
inline std::string Utf8ToGbkOnWin32(const char *src_str)
{
	int len = MultiByteToWideChar(CP_UTF8, 0, src_str, -1, NULL, 0);
	wchar_t* wszGBK = new wchar_t[len + 1];
	memset(wszGBK, 0, len * 2 + 2);
	MultiByteToWideChar(CP_UTF8, 0, src_str, -1, wszGBK, len);
	len = WideCharToMultiByte(CP_ACP, 0, wszGBK, -1, NULL, 0, NULL, NULL);
	char* szGBK = new char[len + 1];
	memset(szGBK, 0, len + 1);
	WideCharToMultiByte(CP_ACP, 0, wszGBK, -1, szGBK, len, NULL, NULL);
	std::string strTemp(szGBK);
	if (wszGBK) delete[] wszGBK;
	if (szGBK) delete[] szGBK;
	return strTemp;
}
#endif // _MSC_VER

/**
* @brief convert unsigned long to string
* @param[in] "unsigned long" the source val
* @return std::string Converted string
*/
inline std::string ulongToUtf8(unsigned long val) {
	char utf8_buf[8] = "";
	size_t val_size = sizeof(val) / sizeof(char);
	unsigned char tmp_val = 0;
	int j = 0;
	for (size_t i = 0; i < val_size; i++)
	{
		tmp_val = (val >> ((val_size - i - 1)*8)) & 0xff;
		if (0 != tmp_val)
		{
			utf8_buf[j++] = tmp_val;
		}
	}

	return utf8_buf;
}

/**
* @brief set windows console to utf-8 encode
*/
inline void initConsole()
{
#ifdef _MSC_VER
	system("CHCP 65001"); /**< utf-8 code */
	CONSOLE_FONT_INFOEX info = { 0 };
	info.cbSize = sizeof(info);
	info.dwFontSize.Y = 16;
	info.FontWeight = FW_NORMAL;
	wcscpy(info.FaceName, L"Consolas");
	SetCurrentConsoleFontEx(GetStdHandle(STD_OUTPUT_HANDLE), NULL, &info);
#else
#endif // _MSC_VER

}

QPANDA_END