#include <algorithm>
#include <iostream>
#include <stdexcept>
#include "QString.h"

namespace QPanda
{

	QString::QString()
	{

	}

	std::vector<QString> QString::split(const QString &sep, QString::SplitBehavior behavior) const
	{
		std::string delimiters = sep.data();
		std::vector<QString> result;

		size_t current;
		size_t next = -1;
		do
		{
			if (behavior == SkipEmptyParts)
			{
				next = m_data.find_first_not_of(delimiters, next + 1);
				if (next == std::string::npos)
				{
					break;
				}

				next -= 1;
			}

			current = next + 1;
			next = m_data.find_first_of(delimiters, current);
			result.emplace_back(m_data.substr(current, next - current));
		} while (next != std::string::npos);

		return result;
	}

	QString QString::trimmed() const
	{
		const std::string delimiters = " \f\n\r\t\v";
		std::string tmp_str = m_data.substr(0, m_data.find_last_not_of(delimiters) + 1);

		return QString(tmp_str.substr(tmp_str.find_first_not_of(delimiters)));
	}

	QString QString::toUpper() const
	{
		std::string str = m_data;
		std::transform(str.begin(), str.end(), str.begin(), ::toupper);

		return str;
	}

	QString QString::toLower() const
	{
		std::string str = m_data;
		std::transform(str.begin(), str.end(), str.begin(), ::tolower);

		return str;
	}

	int QString::toInt(bool *ok, BaseCovert base) const
	{
		int value = 0;
		bool flag = false;
		try
		{
			switch (base)
			{
			case BIN:
				value = std::stoi(m_data, nullptr, 2);
				break;
			case DEC:
				value = std::stoi(m_data, nullptr, 10);
				break;
			case HEX:
				value = std::stoi(m_data, nullptr, 16);
				break;
			default:
				break;
			}

			flag = true;
		}
		catch (const std::invalid_argument& ia)
		{
			std::cerr << "Invalid argument: " << ia.what() << std::endl;
			flag = false;
		}
		catch (const std::out_of_range& oor) {
			std::cerr << "Out of Range error: " << oor.what() << std::endl;
			flag = false;
		}

		if (ok)
		{
			*ok = flag;
		}

		return value;
	}

}
