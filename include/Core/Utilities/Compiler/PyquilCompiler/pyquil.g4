grammar pyquil;

prog: declare+ code_block+
;

code_block: loop
    |operation+
;

loop: loop_start 
    operation* 
    loop_if_continue 
    JUMP_LINE
    loop_end;
loop_start:LABEL_LINE;
loop_end:LABEL_LINE;
LABEL_LINE:'LABEL' SPACES LABELNAME NEWLINE;
JUMP_LINE:'JUMP' SPACES LABELNAME NEWLINE;
loop_if_continue:'JUMP-UNLESS' SPACES LABELNAME SPACES bool_val NEWLINE;


operation:measure
    |gate
    |move
    |sub
    ;
declare: 'DECLARE' SPACES var_name SPACES var_mem NEWLINE;
measure: 'MEASURE' SPACES qbit SPACES array_item NEWLINE;
move:'MOVE' SPACES array_item SPACES expr NEWLINE;
sub:'SUB' SPACES array_item SPACES expr NEWLINE;
var_name: STRING;
var_mem: ('BIT'|'REAL'|'OCTET'|'INTEGER') '[' idx ']';
qbit: INT;



gate: 
    MODIFY_OP+ gate
    |GATE1Q SPACES qbit NEWLINE
    |GATE2Q SPACES qbit SPACES qbit NEWLINE
    |GATE3Q SPACES qbit SPACES qbit SPACES qbit NEWLINE
    |GATE1Q1P '(' param ')' SPACES qbit NEWLINE
    |GATE2Q1P '(' param ')' SPACES qbit SPACES qbit NEWLINE
    ;
MODIFY_OP:('DAGGER'|'CONTROLLED' SPACES);
GATE1Q: 'I'|'X'|'Y'|'Z'|'H'|'S'|'T';//单比特无参门
GATE2Q: 'CZ'|'CNOT'|'SWAP'|'ISWAP';//双比特无参门
GATE3Q:'CSWAP';//三比特门
GATE1Q1P:'PHASE';//单比特单参门
GATE2Q1P:'XY'|'CPHASE00'|'CPHASE'|'CPHASE01'|'CPHASE10'
    |'PSWAP';//双比特单参门


bool_val:INT
    |array_item;//整型数组
param: expr;

// 规则：匹配算术表达式  
expr : '-' expr  //匹配负号
    |expr ('*' | '/') expr  
     | expr ('+' | '-') expr  
     | INT  
     | FLOAT  
     | array_item
     | '(' expr ')'  
     | expr '^' expr  // 乘方运算 
     ;

array_item: arrayname '[' idx ']';
arrayname: STRING;
LABELNAME:'@' [a-zA-Z_]+('-'|[a-zA-Z0-9_])*;
STRING: [a-zA-Z_][a-zA-Z0-9_]*;
idx : INT;
FLOAT : [0-9]+ '.' [0-9]+ ;// 规则：匹配浮点数 
INT	: [0-9]+;// 规则：匹配整数 
SPACES: [ \t]+;//匹配一个或多个空格或制表符
NEWLINE: '\r'? '\n'|'\r';// 匹配换行符，兼容Unix/Linux和Windows风格
WS : [ \t]+ -> skip ;// 跳过空格  
JUMP_UNLESS:'JUMP-UNLESS';


