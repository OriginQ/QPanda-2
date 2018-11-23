'''
Test Utils\n
Copyright (C) Origin Quantum 2017-2018\n
Licensed Under Apache Licence 2.0
'''

def test_begin_str(test_name,max_len=40):
    """
    Use with test_end_str(test_name) to make string like this:\n
    ******** Test Begin ********\n
    ******** Test End **********
    """
    if (len(test_name)%2==1):
        star_len=(max_len-len(test_name))//2+1
        return '*'*star_len+" "+test_name+" Begin "+'*'*(star_len-1)
    else:
        star_len=(max_len-len(test_name))//2
        return '*'*star_len+" "+test_name+" Begin "+'*'*star_len
    
def test_end_str(test_name,max_len=40):
    """
    Use with test_begin_str(test_name) to make string like this:\n
    ******** Test Begin ********\n
    ******** Test End **********
    """
    if (len(test_name)%2==1):
        star_len=(max_len-len(test_name))//2+1
        return '*'*star_len+" "+test_name+" End **"+'*'*(star_len-1)
    else:
        star_len=(max_len-len(test_name))//2
        return '*'*star_len+" "+test_name+" End **"+'*'*star_len