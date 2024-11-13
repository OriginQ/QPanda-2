# This code is part of pyqpanda.
#
# (C) Copyright Origin Quantum 2023
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

""" Origin Quantum Computing Cloud Service ToolKit."""

import os
import json
import time
import requests

from typing import List, Dict, Tuple
from pyqpanda import QCloudService, QProg, QVec, NoiseModel, real_chip_type

import logging
from logging import FileHandler
from datetime import date

from .QCloudPlot import show_chip_topology, get_realtime_topology

class QCloud(QCloudService):
    """
    Quantum Computing Cloud Service Utility Class.

    The primary function of this utility class for the Origin Quantum Computing Cloud Service is to package and send quantum circuits to the remote computing service (Quantum Cloud). 
    It then queries the computation results through polling, supporting various simulators and real hardware.

    User API key authentication is required for computations. Please obtain it here : http://qcloud.originqc.com.cn/
    """

    from enum import Enum, auto
    class CloudQMchineType(Enum):

        Full_AMPLITUDE = 0
        NOISE_QMACHINE = 1
        PARTIAL_AMPLITUDE = 2
        SINGLE_AMPLITUDE = 3
        CHEMISTRY = 4
        REAL_CHIP = 5
        QST = 6
        FIDELITY = 7
    
    class TaskStatus(Enum):
        WAITING = 1
        COMPUTING = 2
        FINISHED = 3
        FAILED = 4

    def update_pqc_keys(self,
                        sym_iv : str,
                        sym_key : str):
        
        self.iv = sym_iv
        self.sym_key = sym_key
        self.pqc_init_completed = True

    def show_chip_topology(self,chip_id : int = 72):
        show_chip_topology(chip_id)
    
    def get_realtime_topology(self,
                              chip_id: int,
                              verbose=False):
        get_realtime_topology(chip_id, verbose)
        
    def get_pqc_encryption(self):

        pqc_init_url  = super().pqc_init_url
        pqc_data = self._request_get(pqc_init_url)

        data = json.loads(pqc_data)

        if "success" in data and data["success"]:
            pk0_value = data["obj"]["pk0"]
        else:
            raise RuntimeError("pqc_init Error: " +  data["message"])

        cipher0, cipher1, sym_key, iv = super().enc_hybrid(pk0_value, self.pqc_random_seed)

        keys_data = {
            "cipher1": cipher0,
            "cipher2": cipher1,
            "pkId": data["obj"]["pkId"]
        }

        pqc_keys_url  = super().pqc_keys_url
        pqc_init_message = self._request_post(pqc_keys_url, json.dumps(keys_data))

        data = json.loads(pqc_init_message)
        if "success" in data and not data["success"]:
            raise RuntimeError("pqc_init enc_hybrid Error: " +  data["message"])

        self.log_debug("pqc_init success.")
        self.log_debug("pqc kyber iv : " , iv)
        self.log_debug("pqc kyber sym_key : " , sym_key)
            
        return iv, sym_key
        # self.iv = iv
        # self.sym_key = sym_key
        # self.pqc_init_completed = True

    from typing import Union
    def _random_num_process(self, data: Union[bytes, str]) -> str:
        
        if isinstance(data, bytes):
            if len(data) > 96:
                data = data[:96]
            elif len(data) < 96:
                data += b'\x00' * (96 - len(data))
            processed_data = data.hex()
            
        elif isinstance(data, str):
            
            if all(c in '0123456789abcdefABCDEF' for c in data):
                if len(data) > 192:
                    data = data[:192]
                elif len(data) < 192:
                    data += '0' * (192 - len(data))
                processed_data = data
            else:
                raise ValueError("Provided string is not a valid hexadecimal string")
        else:
            raise TypeError("Input must be either bytes or string")
        
        return processed_data
    
    def setup_logger(self):
        logger = logging.getLogger('QCloud')
        logger.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

        if self.log_to_console:
            
            console_handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)

        if not self.log_to_console:
            
            today = date.today()
            filename = f"QCloud-{today}.log"
    
            file_handler = FileHandler(filename)
            file_handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        return logger
    
    def init_qvm(self, 
                 user_token: str, 
                 enable_logging: bool = False,
                 log_to_console: bool = True,
                 use_bin_or_hex = True,
                 enable_pqc_encryption = False,
                 random_num : Union[bytes, str] = os.urandom(96),
                 request_time_out = 100):
        
        """
        init quantum virtual machine
        """
        
        #log setting
        self.enable_logging = enable_logging
        self.log_to_console = log_to_console
        self.logger = self.setup_logger() if enable_logging else None

        #pqc setting
        self.pqc_init_completed = False
        self.pqc_random_seed = self._random_num_process(random_num)  
        self.enable_pqc_encryption = enable_pqc_encryption

        #qcloud task setting
        self.use_bin_or_hex = use_bin_or_hex
        
        #https setting
        self.headers = {
            # 'Transfer-Encoding': 'chunked',
            'origin-language': 'en',
            'Connection': 'keep-alive',
            'Content-Type': 'application/json;charset=UTF-8',
            'Authorization': 'oqcs_auth=' + user_token
        }
        
        self.request_time_out = request_time_out
        
        super().init(user_token)

    def log_debug(self, *args):
        
        if self.enable_logging:
            message = ' '.join(map(str, args))
            log_method = getattr(self.logger, "debug")
            log_method(message)
            
    def log_error(self, *args):
            
        if self.enable_logging:
            message = ' '.join(map(str, args))
            log_method = getattr(self.logger, "error")
            log_method(message)

    def convert_result_format(self, 
                              input_dict: dict,
                              binary_size : int):
        
        """
        convert result format from binary to hex 

        Args:
            input_dict (dict): origin result dict
            use_bin_or_hex (bool): True -> use binary result format, False -> Hex

        Returns:
            result: dict.
        """

        if(self.use_bin_or_hex): 

            # convert from hex -> bin
            def hex_to_binary(hex_str, binary_length):
                if hex_str.startswith('0X'):
                    hex_str = hex_str[2:]
                binary_str = bin(int(hex_str, 16))[2:]
                return binary_str.zfill(binary_length)
            
            if(self.enable_pqc_encryption):
                result_dict = {}
                for key, value in input_dict.items():
                    result_dict[hex_to_binary(key, binary_size)] = value
                return result_dict
            
            else:
                result_dict = {}
                for key, value in input_dict.items():
                    key = key.upper()
                    if key.startswith('0X'):  # Check if the key is a hex string
                        result_dict[hex_to_binary(key, binary_size)] = value
                    else:
                        result_dict[key] = value
                
                return result_dict

        else:
            
            if(self.enable_pqc_encryption):
                return input_dict

            # convert from bin -> hex
            def binary_to_hex(binary_str):
                return hex(int(binary_str, 2))

            result_dict = {}
            for key, value in input_dict.items():
                if all(c in '01' for c in key):  # Check if the key is a binary string
                    result_dict[binary_to_hex(key)] = value
                else:
                    result_dict[key] = value

            return result_dict
        
    def _request_get(self,
                     get_url : str = None,
                     params = None, 
                     extra_headers: dict = None):
        """
        Send a GET request with optional query parameters and headers.

        Args:
            url (str): The URL to send the GET request to.
            params (dict, optional): Query parameters to include in the request.
            headers (dict, optional): Headers to include in the request.

        Returns:
            str: Response content if the request is successful, otherwise an error message.

        Raises:
            Exception: If the request fails.
        """

        if extra_headers:
            self.headers.update(extra_headers)

        try:
                        
            self.log_debug("get headers : ", json.dumps(self.headers))
            self.log_debug("get url : ", get_url)
            self.log_debug("get data : ", json.dumps(params))
        
            # Send GET request with optional parameters and headers
            response = requests.get(get_url, params=params, headers=self.headers)

            # Check response status code
            response.raise_for_status()

            # Return response content if request is successful
            return response.text
        
        except requests.exceptions.RequestException as e:
            self.log_error("get error : ", str(e))
            raise RuntimeError("Error: " + str(e))
        
    def _request_post(self, 
                      str_url : str = None,
                      post_data : str = None,
                      use_json_format : bool = True,
                      extra_headers: dict = None):
        """
        Sends a post request to the specified URL with JSON data.

        Args:
            str_url (str): The URL to send the request to.
            post_json (str): JSON data to be sent in the request body.
            extra_headers (dict): Additional HTTP headers to be included.

        Returns:
            str: Response text from the server.

        Raises:
            RuntimeError: If the request fails.
        """   

        if extra_headers:
            self.headers.update(extra_headers)

        timeout = 100
        if self.request_time_out is not None:
            timeout = self.request_time_out
            
        self.log_debug("post headers : ", json.dumps(self.headers))
        self.log_debug("post url : ", str_url)
        self.log_debug("post data : ", post_data)

        if use_json_format:
            json_dict = json.loads(post_data)
            response = requests.post(url = str_url, 
                                    headers = self.headers, 
                                    json = json_dict, 
                                    verify = False, 
                                    timeout = timeout)
        else:
            response = requests.post(url = str_url, 
                                     headers = self.headers, 
                                     data = post_data, 
                                     verify = False, 
                                     timeout = timeout)
        
        # response check
        if response.status_code == 200:
            self.log_debug("post response : ", response.text)
            return response.text
        else:
            self.log_error("error", "post response : ", response.text)
            raise RuntimeError("request failed")
        
    def _query_result_json(self, taskid : str):
        """
        Queries the result of a computation task using task ID.

        Args:
            taskid (str): The task ID to query.

        Returns:
            str: Result string of the computation.

        Notes:
            This function polls the server until the computation is complete.
        """
        
        result_recv_string = None

        while True:

            time.sleep(1)

            obj = {"taskId": taskid,
                   "apiKey": self.user_token}

            response_string = self._request_post(super().inquire_url, json.dumps(obj))
            is_retry_again, query_recv_string = super().cyclic_query(response_string)

            if not is_retry_again:
                result_recv_string = query_recv_string
                break 

        return result_recv_string
       
    def _pqc_batch_query(self, recv_json):
    
        recv_doc = json.loads(recv_json)

        if not recv_doc["success"]:
            raise Exception("recv json parsed failed : ", recv_doc["message"])
        
        # pqc decode 
        task_data = recv_doc["obj"]
        task_status = int(task_data['taskStatus'])

        # CloudQMchineType only in REAL_CHIP    
        if task_status == self.TaskStatus.FINISHED.value:

            measure_qubits_num = []

            for qubit_size in task_data["measureQubitSize"]:
                qubit_addr = int(qubit_size)
                measure_qubits_num.append(qubit_addr)

            result_string = super().sm4_decode(self.sym_key, 
                                               self.iv, 
                                               recv_doc["obj"]["taskResult"])
            
            result_string_array = []
            for item in json.loads(result_string):
                result_string_array.append(json.dumps(item))

            origin_result = super().query_prob_dict_result_batch(result_string_array)

            batch_result = []
            for i in range(len(origin_result)):
                batch_result.append(self.convert_result_format(origin_result[i], measure_qubits_num[i]))

            return [task_status, batch_result]
                      
        elif task_status == self.TaskStatus.FAILED.value:
            error_message = task_data.get("errorMessage", "wrong status : unknown error")
            raise Exception("Task execution failed", error_message)
        
        else:
            return [task_status , []]
    
    def _pqc_batch_query_result_json(self, taskid : str):
        """
        Queries the pqc batch task result of a computation task using task ID.

        Args:
            taskid (str): The task ID to query.

        Returns:
            str: Result string of the computation.

        Notes:
            This function polls the server until the computation is complete.
        """
        
        while True:

            time.sleep(1)

            obj = {"taskId": taskid}

            response_string = self._request_post(super().pqc_batch_inquire_url, json.dumps(obj))
            task_status, final_result = self._pqc_batch_query(response_string)

            if task_status== self.TaskStatus.FINISHED.value:
                return final_result;
    
    def _batch_query_result_json(self, taskid : str):
        """
        Queries the result of a batch computation task using task ID.

        Args:
            taskid (str): The task ID to query.

        Returns:
            List[str]: List of result strings for individual computations.

        Notes:
            This function polls the server until the batch computation is complete.
        """   
        
        result_str_list = None

        while True:

            time.sleep(1)

            obj = {"taskId": taskid}

            response_string = self._request_post(super().batch_inquire_url, json.dumps(obj))
            is_retry_again, query_recv_string = super().batch_cyclic_query(response_string)

            if not is_retry_again:
                result_str_list = query_recv_string
                break 

        return result_str_list
    
    def _submit_and_query(self, task_msg : str):
        """
        Submits a computation task and queries the result.

        Args:
            task_msg (str): Message containing the computation task details.

        Returns:
            str: Result JSON of the computation.

        Notes:
            This function combines submission and result querying.
        """

        submit_url = super().compute_url
        
        recv_string = self._request_post(submit_url, task_msg)
        task_id = super().parse_get_task_id(recv_string)

        result_json = self._query_result_json(task_id)

        return result_json
    
    def _pqc_batch_submit_and_query(self, 
                                    task_msg : str):
        """
        Submits a pqc computation batch task and queries the result.

        Args:
            task_msg (str): Message containing the computation task details.

        Returns:
            str: Result JSON of the computation.

        Notes:
            This function combines submission and result querying.
        """

        pqc_batch_submit_url = super().pqc_batch_compute_url
        
        recv_string = self._request_post(pqc_batch_submit_url, task_msg)
        task_id = super().parse_get_task_id(recv_string)

        return self._pqc_batch_query_result_json(task_id)

    def _batch_submit_and_query(self, 
                                task_msg : str,
                                batch_url : str,
                                extar_header_data : dict = None):
        """
        Submits a batch computation task and queries the result.

        Args:
            task_msg (str): Message containing the batch computation task details.

        Returns:
            List[str]: List of result JSONs for individual computations in the batch.

        Notes:
            This function combines submission and result querying for batch computations.
        """
      
        if extar_header_data:
            recv_string = self._request_post(str_url=batch_url, 
                                             post_data=task_msg,
                                             use_json_format=False,
                                             extra_headers=extar_header_data)
        else:
            recv_string = self._request_post(batch_url, task_msg)

        task_id = super().parse_get_task_id(recv_string)

        result_json = self._batch_query_result_json(task_id)

        return result_json
    
    def query_task_state_result(self, task_id : str, is_real_chip_task : bool = True):
    
        obj = {"taskId": task_id}
        
        inquire_url = super().inquire_url

        # enable pqc encryption
        if self.enable_pqc_encryption and is_real_chip_task:
            
            if not self.pqc_init_completed:
                raise RuntimeError("To use PQC encryption, "
                                   "need to first call get_pqc_encryption() obtain the key, "
                                   "and then call update_pqc_keys() to update and store the key")
                
            inquire_url = super().pqc_inquire_url
        
        # disable pqc encryption
        recv_json = self._request_post(inquire_url, json.dumps(obj))

        recv_dict = json.loads(recv_json)

        if not recv_dict["success"]:
            message = recv_dict["message"]
            raise Exception(f"query task error : {message}")

        result_list = recv_dict["obj"]["qcodeTaskNewVo"]["taskResultList"]

        status = int(result_list[0]["taskState"])
        backend_type = int(result_list[0]["rQMachineType"])

        if status == self.TaskStatus.FINISHED.value:  # TaskStatus::FINISHED

            measure_qubits_num = []

            if "measureQubitSize" not in result_list[0]:
                raise RuntimeError("measureQubitSize not found")

            for qubit_size in result_list[0]["measureQubitSize"]:
                qubit_addr = int(qubit_size)
                measure_qubits_num.append(qubit_addr)

            # CloudQMchineType::REAL_CHIP, NOISE_QMACHINE, Full_AMPLITUDE, PARTIAL_AMPLITUDE, SINGLE_AMPLITUDE
            if backend_type in [self.CloudQMchineType.REAL_CHIP.value]:  
                
                if self.enable_pqc_encryption and self.pqc_init_completed:
                    result_string = super().sm4_decode(self.sym_key, self.iv, result_list[0]["taskResult"])
                else:
                    result_string = result_list[0]["taskResult"]

                origin_result = super().query_prob_dict_result(result_string)
                result = self.convert_result_format(origin_result, measure_qubits_num[0])
                return [status, result]
            
            if backend_type in [self.CloudQMchineType.Full_AMPLITUDE.value,
                                self.CloudQMchineType.NOISE_QMACHINE.value]:  
                result_string = result_list[0]["taskResult"]

                origin_result = super().query_prob_dict_result(result_string)
                return [status, origin_result]
            
            if backend_type ==  self.CloudQMchineType.PARTIAL_AMPLITUDE.value:

                result_string = result_list[0]["taskResult"]
                origin_result = super().query_state_dict_result(result_string)
                return [status, origin_result]
            
            if backend_type ==  self.CloudQMchineType.SINGLE_AMPLITUDE.value:
                result_string = result_list[0]["taskResult"]
                return [status, super().query_comolex_result(result_string)]

            elif backend_type == self.CloudQMchineType.QST.value:
                result_string = result_list[0]["qstresult"]
                return [status, super().query_qst_result(result_string)]

            elif backend_type == self.CloudQMchineType.FIDELITY.value:
                fidelity_value = float(result_list[0]["qstfidelity"])
                result_string = json.dumps({"value": fidelity_value})

                return [status, fidelity_value]

            else:
                raise Exception("Wrong backend type")

        elif status == self.TaskStatus.FAILED.value:
            if "errorMessage" in result_list[0]:
                error_message = result_list[0]["errorMessage"]
            else:
                error_message = "Unknown error"

            self.log_error("error", "Task process error : ", error_message)
            return [status, {}]

        else:
            return [status, {}]
        
    def query_batch_task_state_result(self,
                                      task_id : str):
        
        obj = {"taskId": task_id}
        
        # enable pqc encryption
        if self.enable_pqc_encryption:

            if not self.pqc_init_completed:
                raise RuntimeError("To use PQC encryption, "
                                   "need to first call get_pqc_encryption() obtain the key, "
                                   "and then call update_pqc_keys() to update and store the key")

            response_string = self._request_post(super().pqc_batch_inquire_url, json.dumps(obj))    
            return self._pqc_batch_query(response_string)
    
        # disable pqc encryption
        recv_json = self._request_post(super().batch_inquire_url, json.dumps(obj))
         
        recv_dict = json.loads(recv_json)

        if not recv_dict["success"]:
            message = recv_dict["message"]
            raise Exception(f"query task error : {message}")

        state_str = recv_dict["obj"]["taskStatus"]
        status = int(state_str)

        if status == self.TaskStatus.FINISHED.value:  # TaskStatus::FINISHED

            measure_qubits_num = []

            for qubit_size in recv_dict["obj"]["measureQubitSize"]:
                qubit_addr = int(qubit_size)
                measure_qubits_num.append(qubit_addr)

            result_array = [result_string for result_string in recv_dict["obj"]["taskResult"]]
            
            origin_result = super().query_prob_dict_result_batch(result_array)

            result = []
            for i in range(len(origin_result)):
                result.append(self.convert_result_format(origin_result[i], measure_qubits_num[i]))

            return [status, result]

        elif status == self.TaskStatus.FAILED.value:

            if "errorMessage" in recv_dict["obj"]:
                error_message = recv_dict["obj"]["errorMessage"]
            else:
                error_message = "unknown error"

            self.log_error("error", "Task process error : ", error_message)
            return [status, {}]

        else:
            return [status, {}]
        
    def estimate_price(self,
                       qubit_num : int,
                       shot : int,
                       qprogCount : int = 1,
                       epoch : int = 1):
        
        data = {
            "qubitNum": qubit_num,
            "shot": shot,
            "qprogCount": qprogCount,
            "epoch": epoch
        }

        json_string = json.dumps(data)
        recv_string = self._request_post(super().estimate_url, json_string)

        doc = json.loads(recv_string)

        if not doc["success"]:
            message = doc["message"]
            raise RuntimeError(message)
        else:
            estimate_value_str = doc["obj"]
            return float(estimate_value_str)

    def async_full_amplitude_measure(self, 
                                     prog: QProg, 
                                     shot: int, 
                                     task_name: str = 'QPanda Experiment'):
        """
        Execute a full amplitude measurement on the Quantum Cloud Service.

        Args:
            prog (QProg): Quantum program containing the circuit to be measured.
            shot (int): Number of measurements to perform.
            task_name (str, optional): Task name for identification. Defaults to 'QPanda Experiment'.

        Returns:
            Task_id[str]: A task id for current task
        """
        super().build_init_object(prog, task_name)
        task_msg = super().build_full_amplitude_measure(shot)

        recv_string = self._request_post(super().compute_url, task_msg)
        return super().parse_get_task_id(recv_string)

    def full_amplitude_measure(self, 
                               prog: QProg, 
                               shot: int, 
                               task_name: str = 'QPanda Experiment'):
        """
        Execute a full amplitude measurement on the Quantum Cloud Service.

        Args:
            prog (QProg): Quantum program containing the circuit to be measured.
            shot (int): Number of measurements to perform.
            task_name (str, optional): Task name for identification. Defaults to 'QPanda Experiment'.

        Returns:
            Dict[str, float]: Dictionary containing probabilities of measurement outcomes.
        """
        super().build_init_object(prog, task_name)
        task_msg = super().build_full_amplitude_measure(shot)

        result_json = self._submit_and_query(task_msg)

        origin_result = super().query_prob_dict_result(result_json)
        return origin_result
    
    def async_full_amplitude_pmeasure(self, 
                                     prog: QProg, 
                                     qvec: List[int], 
                                     task_name: str = 'QPanda Experiment'):
        """
        Execute a full amplitude probability measurement on the Quantum Cloud Service.

        Args:
            prog (QProg): Quantum program containing the circuit to be measured.
            shot (int): Number of measurements to perform.
            task_name (str, optional): Task name for identification. Defaults to 'QPanda Experiment'.

        Returns:
            Task_id[str]: A task id for current task
        """
        super().build_init_object(prog, task_name)
        task_msg = super().build_full_amplitude_pmeasure(qubit_vec=qvec)

        recv_string = self._request_post(super().compute_url, task_msg)
        return super().parse_get_task_id(recv_string)

    def full_amplitude_pmeasure(self, 
                                prog: QProg, 
                                qvec: List[int], 
                                task_name: str = 'QPanda Experiment'):
        """
        Execute a full amplitude probability measurement on the Quantum Cloud Service.

        Args:
            prog (QProg): Quantum program containing the circuit to be measured.
            qvec (List[int]): List of qubits to be measured.
            task_name (str, optional): Task name for identification. Defaults to 'QPanda Experiment'.

        Returns:
            Dict[str, float]: Dictionary containing probabilities of measurement outcomes.
        """
        super().build_init_object(prog, task_name)
        task_msg = super().build_full_amplitude_pmeasure(qvec)

        result_json = self._submit_and_query(task_msg)

        origin_result = super().query_prob_dict_result(result_json)
        return origin_result

    def get_expectation(self, 
                        prog: QProg, 
                        hamiltonian: List[Tuple[Dict[int,str],float]], 
                        qvec: QVec, 
                        task_name: str = 'QPanda Experiment'):
        """
        Calculate the expectation value of a Hamiltonian on the Quantum Cloud Service.

        Args:
            prog (QProg): Quantum program containing the circuit for state preparation.
            hamiltonian (List[Tuple[Dict[int, str], float]]): List of terms in the Hamiltonian along with their coefficients.
            qvec (QVec): List of qubits representing the quantum state.
            task_name (str, optional): Task name for identification. Defaults to 'QPanda Experiment'.

        Returns:
            float: Expectation value of the Hamiltonian.
        """
        super().build_init_object(prog, task_name)
        task_msg = super().build_get_expectation(hamiltonian, qvec)

        result_json = self._submit_and_query(task_msg)

        return super().query_prob_result(result_json)

    def get_state_fidelity(self, 
                           prog: QProg, 
                           shot: int, 
                           chip_id: int = 2, 
                           is_amend: bool = True, 
                           is_mapping: bool = True, 
                           is_optimization: bool = True,
                           task_name: str = 'QPanda Experiment'):
        """
        Get the state fidelity of the Quantum Real Chip.

        Args:
            prog (QProg): Quantum program containing the circuit for state preparation.
            shot (int): Number of measurements to perform.
            chip_id (int, optional): ID of the quantum chip. Defaults to 2.
            is_amend (bool, optional): Flag for amplitude amplification. Defaults to True.
            is_mapping (bool, optional): Flag for qubit mapping. Defaults to True.
            is_optimization (bool, optional): Flag for gate optimization. Defaults to True.
            task_name (str, optional): Task name for identification. Defaults to 'QPanda Experiment'.

        Returns:
            float: State fidelity value.
        """
        super().build_init_object(prog, task_name)
        task_msg = super().build_get_state_fidelity(shot=shot,
                                                    chip_id=chip_id,
                                                    is_amend=is_amend,
                                                    is_mapping=is_mapping,
                                                    is_optimization=is_optimization)

        result_json = self._submit_and_query(task_msg)

        return super().query_prob_result(result_json)

    def get_state_tomography_density(self, 
                                     prog: QProg, 
                                     shot: int, 
                                     chip_id: int = 2, 
                                     is_amend: bool = True, 
                                     is_mapping: bool = True, 
                                     is_optimization: bool = True, 
                                     task_name: str = 'QPanda Experiment'):
        """
        Get the density matrix for state tomography on the Quantum Cloud Service.

        Args:
            prog (QProg): Quantum program containing the circuit for state preparation.
            shot (int): Number of measurements to perform.
            chip_id (int, optional): ID of the quantum chip. Defaults to 2.
            is_amend (bool, optional): Flag for amplitude amplification. Defaults to True.
            is_mapping (bool, optional): Flag for qubit mapping. Defaults to True.
            is_optimization (bool, optional): Flag for gate optimization. Defaults to True.
            task_name (str, optional): Task name for identification. Defaults to 'QPanda Experiment'.

        Returns:
            List[List[complex]]: Density matrix representing the quantum state.
        """
        super().build_init_object(prog, task_name)
        task_msg = super().build_get_state_tomography_density(shot=shot,
                                                             chip_id=chip_id,
                                                             is_amend=is_amend,
                                                             is_mapping=is_mapping,
                                                             is_optimization=is_optimization)

        result_json = self._submit_and_query(task_msg)

        return super().query_qst_result(result_json)

    def async_noise_measure(self, 
                            prog: QProg, 
                            shot: int, 
                            task_name: str = 'QPanda Experiment'):
        """
        Measure noise in the quantum computation.

        Args:
            prog (QProg): The quantum circuit to be executed.
            shot (int): The number of shots (measurement repetitions).
            task_name (str, optional): The name of the QPanda Experiment task. Defaults to 'QPanda Experiment'.

        Returns:
            Task_id[str]: A task id for current task
        """

        super().build_init_object(prog, task_name)
        task_msg = super().build_noise_measure(shots=shot)

        recv_string = self._request_post(super().compute_url, task_msg)
        return super().parse_get_task_id(recv_string)

    def noise_measure(self, 
                      prog: QProg, 
                      shot: int, 
                      task_name: str = 'QPanda Experiment'):
        """
        Measure noise in the quantum computation.

        Args:
            prog (QProg): The quantum circuit to be executed.
            shot (int): The number of shots (measurement repetitions).
            task_name (str, optional): The name of the QPanda Experiment task. Defaults to 'QPanda Experiment'.

        Returns:
            Dict[str, float]: A dictionary containing measurement results.
        """

        super().build_init_object(prog, task_name)
        task_msg = super().build_noise_measure(shots=shot)

        result_json = self._submit_and_query(task_msg)

        origin_result = super().query_prob_dict_result(result_json)
        return origin_result

    def async_partial_amplitude_pmeasure(self, 
                                   prog: QProg, 
                                   amp_vec: List[str], 
                                   task_name: str = 'QPanda Experiment'):
        """
        Perform partial amplitude measurement in the quantum computation.

        Args:
            prog (QProg): The quantum circuit to be executed.
            amp_vec (List[str]): List of amplitude vectors to be measured.
            task_name (str, optional): The name of the QPanda Experiment task. Defaults to 'QPanda Experiment'.

        Returns:
            Task_id[str]: A task id for current task
        """

        super().build_init_object(prog, task_name)
        task_msg = super().build_partial_amplitude_pmeasure(amplitudes=amp_vec)

        recv_string = self._request_post(super().compute_url, task_msg)
        return super().parse_get_task_id(recv_string)

    def partial_amplitude_pmeasure(self, 
                                   prog: QProg, 
                                   amp_vec: List[str], 
                                   task_name: str = 'QPanda Experiment'):
        """
        Perform partial amplitude measurement in the quantum computation.

        Args:
            prog (QProg): The quantum circuit to be executed.
            amp_vec (List[str]): List of amplitude vectors to be measured.
            task_name (str, optional): The name of the QPanda Experiment task. Defaults to 'QPanda Experiment'.

        Returns:
            Dict[str, complex]: A dictionary containing complex-valued measurement results.
        """

        super().build_init_object(prog, task_name)
        task_msg = super().build_partial_amplitude_pmeasure(amplitudes=amp_vec)

        result_json = self._submit_and_query(task_msg)

        origin_result = super().query_state_dict_result(result_json)
        return origin_result
    
    def async_real_chip_measure(self, 
                                prog: Union[QProg, str], 
                                shot: int, 
                                chip_id: int = 2, 
                                is_amend: bool = True, 
                                is_mapping: bool = True, 
                                is_optimization: bool = True, 
                                task_name: str = 'QPanda Experiment',
                                task_from = 4):
        """
        Perform measurements on a real quantum chip.

        Args:
            prog (QProg): The quantum circuit to be executed.
            shot (int): The number of shots (measurement repetitions).
            chip_id (int, optional): Identifier for the specific quantum chip. Defaults to 2.
            is_amend (bool, optional): Flag indicating whether to perform amendments. Defaults to True.
            is_mapping (bool, optional): Flag indicating whether to perform qubit mapping. Defaults to True.
            is_optimization (bool, optional): Flag indicating whether to perform optimization. Defaults to True.
            task_name (str, optional): The name of the QPanda Experiment task. Defaults to 'QPanda Experiment'.
            task_from (int): Task source identifier (default is 4, represented by QPanda/pyqpanda).

        Returns:
            taskid[str]: real chip task id
        """

        super().build_init_object(prog, task_name, task_from)
        task_msg = super().build_real_chip_measure(shots=shot,
                                                chip_id=chip_id,
                                                is_amend=is_amend,
                                                is_mapping=is_mapping,
                                                is_optimization=is_optimization)

        # disable pqc encryption
        if not self.enable_pqc_encryption:

            recv_string = self._request_post(super().compute_url, task_msg)
            return super().parse_get_task_id(recv_string)
        
        # enable pqc encryption
        else:
                      
            if not self.pqc_init_completed:
                raise RuntimeError("To use PQC encryption, "
                                   "need to first call get_pqc_encryption() obtain the key, "
                                   "and then call update_pqc_keys() to update and store the key")

            pqc_task_msg = self.pqc_encrypt_and_combine_json(task_msg)
            
            recv_string = self._request_post(super().pqc_compute_url, pqc_task_msg)
            return super().parse_get_task_id(recv_string)
        
    def pqc_encrypt(self, data : str):
        """
        Encrypts the given data using the pqc encryption algorithm.

        Args:
            data (str): The data to be encrypted.

        Returns:
            str: The encrypted data.
        """
        
        encrypted_data = super().sm4_encode(self.sym_key,  self.iv, data) 
        return encrypted_data

    def pqc_encrypt_and_combine_json(self, json_str : str):
        """
        Encrypts the 'code' field and the remaining JSON data, and combines them into a new JSON string.

        Args:
            json_str (str): The original JSON string.

        Returns:
            str: The new JSON string with encrypted 'code' field and remaining JSON data.
        """
        # Parse JSON string into dictionary
        json_data = json.loads(json_str)
        
        # Get the value of 'code' field and encrypt it
        encrypt_code_value = self.pqc_encrypt(json.dumps(json_data["code"]))
        
        # Remove the 'code' field from the dictionary
        json_data["code"] = encrypt_code_value
        
        pqc_json_str = json.dumps(json_data)
        return pqc_json_str
    
    def pqc_encrypt_and_combine_batch_json(self,json_str : str):
        """
        Encrypts the 'code' field and the remaining JSON data, and combines them into a new JSON string.

        Args:
            json_str (str): The original JSON string.

        Returns:
            str: The new JSON string with encrypted 'code' field and remaining JSON data.
        """
        # Parse JSON string into dictionary
        json_data = json.loads(json_str)
        
        encrypt_code_array_value = self.pqc_encrypt(json.dumps(json_data["qprogArr"]))
        
        json_data["qprogStr"] = encrypt_code_array_value
        
        del json_data["qprogArr"]
        
        # Convert the new JSON object to a JSON string
        pqc_json_str = json.dumps(json_data)
        return pqc_json_str

    def real_chip_measure(self, 
                          prog: Union[QProg, str], 
                          shot: int, 
                          chip_id: int = 2, 
                          is_amend: bool = True, 
                          is_mapping: bool = True, 
                          is_optimization: bool = True,
                          task_name: str = 'QPanda Experiment',
                          task_from = 4):
        """
        Perform measurements on a real quantum chip.

        Args:
            prog (QProg): The quantum circuit to be executed.
            shot (int): The number of shots (measurement repetitions).
            chip_id (int, optional): Identifier for the specific quantum chip. Defaults to 2.
            is_amend (bool, optional): Flag indicating whether to perform amendments. Defaults to True.
            is_mapping (bool, optional): Flag indicating whether to perform qubit mapping. Defaults to True.
            is_optimization (bool, optional): Flag indicating whether to perform optimization. Defaults to True.
            task_name (str, optional): The name of the QPanda Experiment task. Defaults to 'QPanda Experiment'.
            task_from (int): Task source identifier (default is 4, represented by QPanda/pyqpanda).

        Returns:
            Dict[str, float]: A dictionary containing measurement results.
        """

        super().build_init_object(prog, task_name,task_from)
        task_msg = super().build_real_chip_measure(shots=shot,
                                                   chip_id=chip_id,
                                                   is_amend=is_amend,
                                                   is_mapping=is_mapping,
                                                   is_optimization=is_optimization)
        
        # disable pqc encryption
        if not self.enable_pqc_encryption:

            result_json = self._submit_and_query(task_msg)
            
            origin_result = super().query_prob_dict_result(result_json)
            return self.convert_result_format(origin_result, super().measure_qubits_num[0])
        
        # enable pqc encryption
        else:
                   
            task_id = self.async_real_chip_measure(prog=prog, 
                                                   shot=shot, 
                                                   chip_id=chip_id, 
                                                   is_amend=is_amend, 
                                                   is_mapping=is_mapping, 
                                                   is_optimization=is_optimization, 
                                                   task_name=task_name,
                                                   task_from=task_from)
            
            while(True):
                time.sleep(1)
                status, result = self.query_task_state_result(task_id)

                if status == QCloud.TaskStatus.FINISHED.value:
                    return result
    
    def async_batch_real_chip_measure(self, 
                                      prog_array: Union[List[QProg], List[str]],
                                      shot: int,
                                      chip_id: real_chip_type = real_chip_type.origin_72, 
                                      is_amend: bool = True, 
                                      is_mapping: bool = True,
                                      is_optimization: bool = True,
                                      batch_id: str = "",
                                      task_from = 4):
        """
        Measure a batch of quantum programs on a real quantum chip.

        Args:
        - prog_array (List[QProg]): List of quantum programs to be executed.
        - shot (int): Number of shots (measurements) to perform for each program.
        - chip_id (real_chip_type, optional): ID of the real quantum chip to use (default is real_chip_type.origin_72).
        - is_amend (bool, optional): Whether to perform amendment on the programs (default is True).
        - is_mapping (bool, optional): Whether to perform qubit mapping (default is True).
        - is_optimization (bool, optional): Whether to perform gate fusion optimization (default is True).
        - task_from (int): Task source identifier (default is 4, represented by QPanda/pyqpanda).
        - batch_id (str): The current batch number information for batch tasks, default to empty

        Returns:
        batch_task_id[str]: batch task id

        Note:
        The function submits a batch of quantum programs for execution on a real quantum chip, 
        retrieves the results, and returns the probabilities of measurement outcomes for each program.

        """

        # if disable pqc encryption
        if not self.enable_pqc_encryption:

            task_msg = super().build_real_chip_measure_batch(prog_list=prog_array,
                                                             shots=shot,
                                                             chip_id=chip_id,
                                                             is_amend=is_amend,
                                                             is_mapping=is_mapping,
                                                             is_optimization=is_optimization,
                                                             enable_compress_check=True,
                                                             batch_id=batch_id,
                                                             task_from=task_from)

            if super().use_compress_data:

                import bz2
                compressed_data = bz2.compress(task_msg.encode('utf-8'))

                configuration_data_str = super().configuration_header_data
                big_data_header = {'configuration' : configuration_data_str}

                recv_str = self._request_post(str_url=super().big_data_batch_compute_url, 
                                              post_data=compressed_data,
                                              use_json_format=False,
                                              extra_headers=big_data_header)
            else:
                recv_str = self._request_post(super().batch_compute_url, task_msg)

            task_id = super().parse_get_task_id(recv_str)

            return task_id
        
        # enable pqc encryption
        else:

            task_msg = super().build_real_chip_measure_batch(prog_list=prog_array,
                                                             shots=shot,
                                                             chip_id=chip_id,
                                                             is_amend=is_amend,
                                                             is_mapping=is_mapping,
                                                             is_optimization=is_optimization,
                                                             enable_compress_check=False,
                                                             batch_id=batch_id,
                                                             task_from=task_from)
                        
            if not self.pqc_init_completed:
                raise RuntimeError("To use PQC encryption, "
                                   "need to first call get_pqc_encryption() obtain the key, "
                                   "and then call update_pqc_keys() to update and store the key")

            pqc_batch_task_msg = self.pqc_encrypt_and_combine_batch_json(task_msg)
            recv_string = self._request_post(super().pqc_batch_compute_url, pqc_batch_task_msg)
            task_id = super().parse_get_task_id(recv_string)
            return task_id

    def batch_real_chip_measure(self, 
                                prog_array: Union[List[QProg], List[str]],
                                shot: int, 
                                chip_id: real_chip_type = real_chip_type.origin_72, 
                                is_amend: bool = True, 
                                is_mapping: bool = True,
                                is_optimization: bool = True,
                                batch_id: str = "",
                                task_from = 4):
        """
        Measure a batch of quantum programs on a real quantum chip.

        Args:
        - prog_array (List[QProg]): List of quantum programs to be executed.
        - shot (int): Number of shots (measurements) to perform for each program.
        - chip_id (real_chip_type, optional): ID of the real quantum chip to use (default is real_chip_type.origin_72).
        - is_amend (bool, optional): Whether to perform amendment on the programs (default is True).
        - is_mapping (bool, optional): Whether to perform qubit mapping (default is True).
        - is_optimization (bool, optional): Whether to perform gate fusion optimization (default is True).
        - task_from (int): Task source identifier (default is 4, represented by QPanda/pyqpanda).
        - batch_id (str): The current batch number information for batch tasks, default to empty
        
        Returns:
        List[Dict[str, float]]: A list of dictionaries containing the probabilities of measurement outcomes for each program.

        Note:
        The function submits a batch of quantum programs for execution on a real quantum chip, 
        retrieves the results, and returns the probabilities of measurement outcomes for each program.

        """

        # if disable pqc encryption
        if not self.enable_pqc_encryption:

            task_msg = super().build_real_chip_measure_batch(prog_list=prog_array,
                                                             shots=shot,
                                                             chip_id=chip_id,
                                                             is_amend=is_amend,
                                                             is_mapping=is_mapping,
                                                             is_optimization=is_optimization,
                                                             enable_compress_check=True,
                                                             batch_id=batch_id,
                                                             task_from=task_from)

            # enable compress data
            if super().use_compress_data:
        
                import bz2
                compressed_data = bz2.compress(task_msg.encode('utf-8'))

                configuration_data_str = super().configuration_header_data
                submit_url = super().big_data_batch_compute_url
                big_data_header = {'configuration' : configuration_data_str}
                result_json = self._batch_submit_and_query(compressed_data, submit_url, big_data_header)
            else:
                submit_url = super().batch_compute_url
                result_json = self._batch_submit_and_query(task_msg, submit_url)
            
            origin_result = super().query_prob_dict_result_batch(result_json)

            result = []
            for i in range(len(origin_result)):
                result.append(self.convert_result_format(origin_result[i], super().measure_qubits_num[i]))

            return result
        
        # enable pqc encryption
        else:

            task_msg = super().build_real_chip_measure_batch(prog_list=prog_array,
                                                             shots=shot,
                                                             chip_id=chip_id,
                                                             is_amend=is_amend,
                                                             is_mapping=is_mapping,
                                                             is_optimization=is_optimization,
                                                             enable_compress_check=False,
                                                             batch_id=batch_id,
                                                             task_from=task_from)
                        
            if not self.pqc_init_completed:
                raise RuntimeError("To use PQC encryption, "
                                   "need to first call get_pqc_encryption() obtain the key, "
                                   "and then call update_pqc_keys() to update and store the key")

        pqc_batch_task_msg = self.pqc_encrypt_and_combine_batch_json(task_msg)
        return self._pqc_batch_submit_and_query(pqc_batch_task_msg)

    def set_noise_model(self, 
                        model: NoiseModel, 
                        single_gate_params: List[float], 
                        single_param_list: List[float]):
        """
        Set the noise model for quantum computation.

        Args:
        - model (NoiseModel): The noise model to be set.
        - single_gate_params (List[float]): List of parameters for single-qubit gates.
        - single_param_list (List[float]): List of parameters for single-qubit gates.

        Returns:
        None
        """
        super().set_noise_model(model,single_gate_params,single_param_list)
        ...

    def set_qcloud_url(self, prefix_url: str):
        """
        Set the QCloud API endpoint.

        Args:
        - prefix_url (str): The prefix URL of the QCloud API.

        Returns:
        None
        """
        super().set_qcloud_url(prefix_url)
        ...

    def async_single_amplitude_pmeasure(self, 
                                        prog: QProg, 
                                        amplitude: str, 
                                        task_name: str = 'QPanda Experiment'):
        """
        Measure the single amplitude of a quantum state.

        Args:
        - prog (QProg): The quantum program containing the state.
        - amplitude (str): The amplitude to measure.
        - task_name (str): Name of the QPanda Experiment task.

        Returns:
            Task_id[str]: A task id for current task
        """
        super().build_init_object(prog, task_name)
        task_msg = super().build_single_amplitude_pmeasure(amplitude=amplitude)

        recv_string = self._request_post(super().compute_url, task_msg)
        return super().parse_get_task_id(recv_string)
    
    def single_amplitude_pmeasure(self, 
                                  prog: QProg, 
                                  amplitude: str, 
                                  task_name: str = 'QPanda Experiment'):
        """
        Measure the single amplitude of a quantum state.

        Args:
        - prog (QProg): The quantum program containing the state.
        - amplitude (str): The amplitude to measure.
        - task_name (str): Name of the QPanda Experiment task.

        Returns:
        complex: The measured amplitude.
        """
        super().build_init_object(prog, task_name)
        task_msg = super().build_single_amplitude_pmeasure(amplitude=amplitude)

        result_json = self._submit_and_query(task_msg)

        return super().query_comolex_result(result_json)

    def pec_error_mitigation(self, 
                             prog: QProg, 
                             shot: int, 
                             expectations: List[str], 
                             chip_id: int = 72, 
                             task_name: str = 'QPanda Experiment'):
        """
        Apply PEC error mitigation to correct measurement errors.

        Args:
        - prog (QProg): The quantum program to apply error mitigation.
        - shot (int): Number of shots for measurements.
        - expectations (List[str]): List of measurement expectations.
        - chip_id (int): Chip identifier.
        - task_name (str): Name of the QPanda Experiment task.

        Returns:
        List[float]: List of corrected expectation values.
        """
        super().build_init_object(prog, task_name)

        from pyqpanda import em_method
        task_msg = super().build_error_mitigation(shots=shot,
                                                  chip_id=chip_id,
                                                  expectations=expectations,
                                                  noise_strength=[],
                                                  qemMethod=em_method.pec)

        result_json = self._submit_and_query(task_msg)

        return super().query_prob_vec_result(result_json)

    def read_out_error_mitigation(self, 
                                  prog: QProg, 
                                  shot: int, 
                                  expectations: List[str], 
                                  chip_id: int = 72, 
                                  task_name: str = 'QPanda Experiment'):
        """
        Apply readout error mitigation to correct measurement errors.

        Args:
        - prog (QProg): The quantum program to apply error mitigation.
        - shot (int): Number of shots for measurements.
        - expectations (List[str]): List of measurement expectations.
        - chip_id (int): Chip identifier.
        - task_name (str): Name of the QPanda Experiment task.

        Returns:
            Dict[str, float]: Dictionary of corrected expectation values.
        """

        super().build_init_object(prog, task_name)

        from pyqpanda import em_method
        task_msg = super().build_read_out_mitigation(shots=shot,
                                                     chip_id=chip_id,
                                                     expectations=expectations,
                                                     noise_strength=[],
                                                     qemMethod=em_method.READ_OUT)

        result_json = self._submit_and_query(task_msg)

        origin_result = super().query_prob_dict_result(result_json)
        return self.convert_result_format(origin_result, super().measure_qubits_num[0])

    def zne_error_mitigation(self, 
                             prog: QProg, 
                             shot: int, 
                             expectations: List[str], 
                             noise_strength: List[float], 
                             chip_id: int = 72, 
                             task_name: str = 'QPanda Experiment'):
        """
        Apply zero-noise extrapolation (ZNE) error mitigation to correct measurement errors.

        Args:
        - prog (QProg): The quantum program to apply error mitigation.
        - shot (int): Number of shots for measurements.
        - expectations (List[str]): List of measurement expectations.
        - noise_strength (List[float]): List of noise strengths.
        - chip_id (int): Chip identifier.
        - task_name (str): Name of the QPanda Experiment task.

        Returns:
        List[float]: List of corrected expectation values.
        """
        super().build_init_object(prog, task_name)

        from pyqpanda import em_method
        task_msg = super().build_error_mitigation(shots=shot,
                                                  chip_id=chip_id,
                                                  expectations=expectations,
                                                  noise_strength=[],
                                                  qemMethod=em_method.zne)

        result_json = self._submit_and_query(task_msg)

        return super().query_prob_vec_result(result_json)
