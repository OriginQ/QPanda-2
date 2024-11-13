'''
Test Utils\n
Copyright (C) Origin Quantum 2017-2018\n
Licensed Under Apache Licence 2.0
'''

def test_begin_str(test_name,max_len=40):
    """
    Generates a string that precedes the beginning of a test case with a specified name,
    ensuring the string is centered within a maximum length defined by `max_len`.

    The output format is:
    <stars> <test_name> Begin <stars>

    If `test_name` has an odd length, an additional star is added to the right side of the
    centered text to maintain symmetry.

        Args:
            test_name (str): The name of the test case to be prefixed.
            max_len (int, optional): The maximum width of the output string. Defaults to 40.

        Returns:
            str: A centered string indicating the beginning of a test case.
    """
    if (len(test_name)%2==1):
        star_len=(max_len-len(test_name))//2+1
        return '*'*star_len+" "+test_name+" Begin "+'*'*(star_len-1)
    else:
        star_len=(max_len-len(test_name))//2
        return '*'*star_len+" "+test_name+" Begin "+'*'*star_len
    
def test_end_str(test_name,max_len=40):
    """
    Generates a formatted string for marking the end of a test case with stars, given a test name.
    
        Args:
            test_name (str): The name of the test case.
            max_len (int, optional): The maximum length of the output string. Defaults to 40 characters.
    
        Returns:
            str: A formatted string with the test name centered and surrounded by stars, indicating the end of the test case.
    
        Usage example:
        >>> test_end_str("MyTest")
        '******** MyTest End **'
    
        Note: 
            This function is intended to be used in conjunction with `test_begin_str(test_name)` for
            creating a complete test case marker.
    """
    if (len(test_name)%2==1):
        star_len=(max_len-len(test_name))//2+1
        return '*'*star_len+" "+test_name+" End **"+'*'*(star_len-1)
    else:
        star_len=(max_len-len(test_name))//2
        return '*'*star_len+" "+test_name+" End **"+'*'*star_len