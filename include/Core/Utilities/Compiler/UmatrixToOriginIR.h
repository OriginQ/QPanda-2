#pragma once
#include <tuple>
#include <Core/QuantumMachine/QVec.h>
#include <Core/QuantumMachine/OriginQuantumMachine.h>
#include "Core/Utilities/Tools/MultiControlGateDecomposition.h"
#include "Core/Utilities/Tools/MatrixDecomposition.h"
#include "Core/Utilities/QProgTransform/TransformDecomposition.h"
#include "Core/Utilities/Compiler/QProgToOriginIR.h"

namespace QPanda {
	
	namespace UTIR {
		//type define, same as that in QPandaNamespace.h
		typedef double qstate_type;
		typedef std::complex <qstate_type> qcomplex_t;
		typedef std::vector <qcomplex_t> QStat;
		typedef std::vector<size_t> Qnum;
		//gate u matrix with info(name,qbit total)
		static std::tuple<std::string, QStat, unsigned int> SQiSW() {
			QStat res = {
				{1,0},{0				,0				  }	,{0					,0				  }	,{0,0},
				{0,0},{ 1 / std::sqrt(2),0				  }	,{0					,1 / std::sqrt(2) }	,{0,0},
				{0,0},{0				,1 / std::sqrt(2) }	,{1 / std::sqrt(2)  ,0				  } ,{0,0},
				{0,0},{0				,0				  }	,{0					,0				  }	,{1,0}
			};
			return { "SQiSW",res,2 };//name,matrix,qbit total
		}

		static std::tuple<std::string, QStat, unsigned int> CSWAP() {
			QStat res = {
				{1,0}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0},
				{0,0}, {1,0}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0},
				{0,0}, {0,0}, {1,0}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0},
				{0,0}, {0,0}, {0,0}, {1,0}, {0,0}, {0,0}, {0,0}, {0,0},
				{0,0}, {0,0}, {0,0}, {0,0}, {1,0}, {0,0}, {0,0}, {0,0},
				{0,0}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0}, {1,0}, {0,0},
				{0,0}, {0,0}, {0,0}, {0,0}, {0,0}, {1,0}, {0,0}, {0,0},
				{0,0}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0}, {0,0}, {1,0}
			};
			return { "CSWAP",res,3 };//name,matrix,qbit total
		}

		static std::tuple<std::string, QStat, unsigned int> ISWAP() {
			QStat res = {
				{1,0}, {0,0}, {0,0}, {0,0},
				{0,0}, {0,0}, {0,1}, {0,0},
				{0,0}, {0,1}, {0,0}, {0,0},
				{0,0}, {0,0}, {0,0}, {1,0}
			};
			return { "ISWAP",res,2 };//name,matrix,qbit total
		}

		static std::tuple<std::string, QStat, unsigned int> PSWAP(double phi) {
			using namespace std;
			QStat res = {
				{1,0}, {0		,0			}, {0		 ,0			}, {0,0},
				{0,0}, {0		,0			}, {cos(phi),sin(phi)	}, {0,0},
				{0,0}, {cos(phi),sin(phi)	}, {0		 ,0			}, {0,0},
				{0,0}, {0		,0			}, {0		 ,0			}, {1,0}
			};
			return { "PSWAP",res,2 };//name,matrix,qbit total
		}

		static std::tuple<std::string, QStat, unsigned int> XY(double phi) {
			using namespace std;
			QStat res = {
				{1,0}, {0			,0				}, {0			 ,0				}, {0,0},
				{0,0}, {cos(phi / 2),0				}, {0			 ,sin(phi / 2)	}, {0,0},
				{0,0}, {0			,sin(phi / 2)	}, {cos(phi / 2) ,0				}, {0,0},
				{0,0}, {0			,0				}, {0			 ,0				}, {1,0}
			};
			return { "XY",res,2 };//name,matrix,qbit total
		}

		static std::tuple<std::string, QStat, unsigned int> SQISW() {
			using namespace std;
			QStat res = {
				{1,0}, {0				,0				}, {0				,0				}, {0,0},
				{0,0}, {1.0 / sqrt(2)	,0				}, {0				,1.0 / sqrt(2)	}, {0,0},
				{0,0}, {0				,1.0 / sqrt(2)	}, {1.0 / sqrt(2)	,0				}, {0,0},
				{0,0}, {0				,0				}, {0				,0				}, {1,0}
			};
			return { "SQISW",res,2 };//name,matrix,qbit total
		}

		static std::tuple<std::string, QStat, unsigned int> FSIM(double theta, double phi) {
			using namespace std;
			QStat res = {
				{1,0}, {0				,0				}, {0				,0				}, {0		,0			},
				{0,0}, {cos(theta / 2)	,0				}, {0				,sin(theta / 2)	}, {0		,0			},
				{0,0}, {0				,sin(theta / 2) }, { cos(theta / 2)	,0				}, {0		,0			},
				{0,0}, {0				,0				}, {0				,0				}, {cos(phi),sin(phi)	}
			};
			return { "FSIM",res,2 };//name,matrix,qbit total
		}

		static std::tuple<std::string, QStat, unsigned int> PHASEDFSIM(double theta, double zeta, double chi, double gamma, double phi) {
			using namespace std;
			QStat res = {
				{1,0}, {0									,0											}, {0									,0											}, {0														,0},
				{0,0}, {cos(gamma + zeta) * cos(theta / 2)	,(0 - sin(gamma + zeta)) * cos(theta / 2)	}, {sin(gamma - chi) * sin(theta / 2)	,cos(gamma - chi) * sin(theta / 2)			}, {0														,0},
				{0,0}, {sin(gamma + chi) * sin(theta / 2)	,cos(gamma + chi) * sin(theta / 2)			}, {cos(gamma - zeta) * cos(theta / 2)	,(0 - sin(gamma - zeta)) * cos(theta / 2)	}, {0														,0},
				{0,0}, {0									,0											}, {0									,0											}, {cos(phi) * cos(2 * gamma) + sin(phi) * sin(2 * gamma)	, sin(phi) * cos(2 * gamma) - cos(phi) * sin(2 * gamma)}
			};
			return { "PHASEDFSIM",res,2 };//name,matrix,qbit total
		}

		static std::tuple<std::string, QStat, unsigned int> PSWAP(const std::vector<double>& params) {
			return PSWAP(params[0]);
		}

		static std::tuple<std::string, QStat, unsigned int> XY(const std::vector<double>& params) {
			return XY(params[0]);
		}


		static std::tuple<std::string, QStat, unsigned int> FSIM(const std::vector<double>& params) {
			return FSIM(params[0], params[1]);
		}

		static std::tuple<std::string, QStat, unsigned int> PHASEDFSIM(const std::vector<double>& params) {
			return PHASEDFSIM(params[0], params[1], params[2], params[3], params[4]);
		}
	

		//Return all the starting indices of string B in string A
		static std::vector<unsigned int> findAllOccurrences(const std::string& A, const std::string& B) {
			std::vector<unsigned int> indices;
			size_t pos = A.find(B);
			while (pos != std::string::npos) {
				indices.push_back(pos);
				pos = A.find(B, pos + 1);
			}
			return indices;
		}
		//Replace all occurrences of the substring starting at each index of string A (with length len) with string B.
		static void replaceSubstrings(std::string& A, const std::string& B,unsigned int len, const std::vector<unsigned int>& indices) {
			std::stringstream ss;
			int idx = 0;
			for (int i = 0; i < A.length();) {
				if (i == indices[idx]&&indices[idx]+len-1<A.length()) {
					for (int j = len; j > 0; j--) {
						i++;
					}
					ss << B;
					idx++;
				}
				else {
					ss << A[i];
					i++;
				}
			}
			A = ss.str();
		}

		//Convert a Western Digital matrix to an OriginIR string without bit declaration statements
		static std::string convert_matrix_to_originir_without_declare(const QStat& u_matrix, const std::vector<unsigned int> qbit_idxs) {
			auto qm = CPUQVM();
			qm.init();
			QVec qbits = qm.qAllocMany(qbit_idxs.size());
			auto cir = matrix_decompose_qr(qbits, u_matrix);
			decompose_multiple_control_qgate(cir, &qm);
			QProg prog(cir);
			std::string originir_str = convert_qprog_to_originir(prog, &qm);
			std::stringstream ss(originir_str), ss2;
			std::string line;
			int declare_row_num = 2;
			while (std::getline(ss, line)) {
				if (declare_row_num > 0) {
					declare_row_num -= 1;
					continue;
				}
				ss2 << line << "\n";
			}
			originir_str = ss2.str();

			//Restore the qbit indices in the pyquil string
			std::unordered_map<int, std::vector<unsigned int>> now_idxs;//Now,the current qbit indices; idxs, all the indices where "now" occurs in the string.
			for (int i = 0; i < qbit_idxs.size(); i++) {
				now_idxs[i] = findAllOccurrences(originir_str, "[" + std::to_string(i) + "]");
			}
			for (int i = 0; i < qbit_idxs.size(); i++) {
				replaceSubstrings(originir_str, "["+std::to_string(qbit_idxs[i])+"]", std::to_string(i).length()+2, now_idxs[i]);
			}

			return originir_str;
		}
	};
};