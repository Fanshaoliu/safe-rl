??
?&?&
:
Add
x"T
y"T
z"T"
Ttype:
2	
W
AddN
inputs"T*N
sum"T"
Nint(0"!
Ttype:
2	??
?
	ApplyAdam
var"T?	
m"T?	
v"T?
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"T?" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
x
Assign
ref"T?

value"T

output_ref"T?"	
Ttype"
validate_shapebool("
use_lockingbool(?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
I
ConcatOffset

concat_dim
shape*N
offset*N"
Nint(0
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
S
DynamicStitch
indices*N
data"T*N
merged"T"
Nint(0"	
Ttype
,
Exp
x"T
y"T"
Ttype:

2
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
9
FloorMod
x"T
y"T
z"T"
Ttype:

2	
=
Greater
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
,
Log
x"T
y"T"
Ttype:

2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
;
Maximum
x"T
y"T
z"T"
Ttype:

2	?
?
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
=
Mul
x"T
y"T
z"T"
Ttype:
2	?
.
Neg
x"T
y"T"
Ttype:

2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
6
Pow
x"T
y"T
z"T"
Ttype:

2	
?
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
PyFunc
input2Tin
output2Tout"
tokenstring"
Tin
list(type)("
Tout
list(type)(?
?
RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	?
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	?
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
?
SplitV

value"T
size_splits"Tlen
	split_dim
output"T*	num_split"
	num_splitint(0"	
Ttype"
Tlentype0	:
2	
N
Squeeze

input"T
output"T"	
Ttype"
squeeze_dims	list(int)
 (
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	
?
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
-
Tanh
x"T
y"T"
Ttype:

2
:
TanhGrad
y"T
dy"T
z"T"
Ttype:

2
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
s

VariableV2
ref"dtype?"
shapeshape"
dtypetype"
	containerstring "
shared_namestring ?
&
	ZerosLike
x"T
y"T"	
Ttype"serve*1.13.12
b'unknown'??
n
PlaceholderPlaceholder*
dtype0*
shape:?????????<*'
_output_shapes
:?????????<
p
Placeholder_1Placeholder*
dtype0*'
_output_shapes
:?????????*
shape:?????????
h
Placeholder_2Placeholder*
shape:?????????*#
_output_shapes
:?????????*
dtype0
h
Placeholder_3Placeholder*#
_output_shapes
:?????????*
shape:?????????*
dtype0
h
Placeholder_4Placeholder*#
_output_shapes
:?????????*
dtype0*
shape:?????????
h
Placeholder_5Placeholder*
shape:?????????*
dtype0*#
_output_shapes
:?????????
h
Placeholder_6Placeholder*
dtype0*#
_output_shapes
:?????????*
shape:?????????
N
Placeholder_7Placeholder*
_output_shapes
: *
dtype0*
shape: 
N
Placeholder_8Placeholder*
shape: *
dtype0*
_output_shapes
: 
?
0pi/dense/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB"<      *"
_class
loc:@pi/dense/kernel
?
.pi/dense/kernel/Initializer/random_uniform/minConst*
dtype0*
valueB
 *?*"
_class
loc:@pi/dense/kernel*
_output_shapes
: 
?
.pi/dense/kernel/Initializer/random_uniform/maxConst*"
_class
loc:@pi/dense/kernel*
dtype0*
valueB
 *>*
_output_shapes
: 
?
8pi/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform0pi/dense/kernel/Initializer/random_uniform/shape*
T0*
dtype0*
_output_shapes
:	<?*
seed2*"
_class
loc:@pi/dense/kernel*

seed 
?
.pi/dense/kernel/Initializer/random_uniform/subSub.pi/dense/kernel/Initializer/random_uniform/max.pi/dense/kernel/Initializer/random_uniform/min*"
_class
loc:@pi/dense/kernel*
T0*
_output_shapes
: 
?
.pi/dense/kernel/Initializer/random_uniform/mulMul8pi/dense/kernel/Initializer/random_uniform/RandomUniform.pi/dense/kernel/Initializer/random_uniform/sub*
T0*
_output_shapes
:	<?*"
_class
loc:@pi/dense/kernel
?
*pi/dense/kernel/Initializer/random_uniformAdd.pi/dense/kernel/Initializer/random_uniform/mul.pi/dense/kernel/Initializer/random_uniform/min*
T0*
_output_shapes
:	<?*"
_class
loc:@pi/dense/kernel
?
pi/dense/kernel
VariableV2*
	container *
_output_shapes
:	<?*"
_class
loc:@pi/dense/kernel*
dtype0*
shape:	<?*
shared_name 
?
pi/dense/kernel/AssignAssignpi/dense/kernel*pi/dense/kernel/Initializer/random_uniform*
validate_shape(*
use_locking(*
_output_shapes
:	<?*"
_class
loc:@pi/dense/kernel*
T0

pi/dense/kernel/readIdentitypi/dense/kernel*
T0*"
_class
loc:@pi/dense/kernel*
_output_shapes
:	<?
?
pi/dense/bias/Initializer/zerosConst*
_output_shapes	
:?*
valueB?*    * 
_class
loc:@pi/dense/bias*
dtype0
?
pi/dense/bias
VariableV2*
_output_shapes	
:?*
dtype0* 
_class
loc:@pi/dense/bias*
shape:?*
shared_name *
	container 
?
pi/dense/bias/AssignAssignpi/dense/biaspi/dense/bias/Initializer/zeros*
_output_shapes	
:?*
T0* 
_class
loc:@pi/dense/bias*
validate_shape(*
use_locking(
u
pi/dense/bias/readIdentitypi/dense/bias* 
_class
loc:@pi/dense/bias*
_output_shapes	
:?*
T0
?
pi/dense/MatMulMatMulPlaceholderpi/dense/kernel/read*
transpose_a( *
transpose_b( *(
_output_shapes
:??????????*
T0
?
pi/dense/BiasAddBiasAddpi/dense/MatMulpi/dense/bias/read*
T0*(
_output_shapes
:??????????*
data_formatNHWC
Z
pi/dense/TanhTanhpi/dense/BiasAdd*
T0*(
_output_shapes
:??????????
?
2pi/dense_1/kernel/Initializer/random_uniform/shapeConst*$
_class
loc:@pi/dense_1/kernel*
_output_shapes
:*
valueB"      *
dtype0
?
0pi/dense_1/kernel/Initializer/random_uniform/minConst*$
_class
loc:@pi/dense_1/kernel*
valueB
 *׳ݽ*
_output_shapes
: *
dtype0
?
0pi/dense_1/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *$
_class
loc:@pi/dense_1/kernel*
dtype0*
valueB
 *׳?=
?
:pi/dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform2pi/dense_1/kernel/Initializer/random_uniform/shape*$
_class
loc:@pi/dense_1/kernel*

seed *
dtype0*
T0*
seed2* 
_output_shapes
:
??
?
0pi/dense_1/kernel/Initializer/random_uniform/subSub0pi/dense_1/kernel/Initializer/random_uniform/max0pi/dense_1/kernel/Initializer/random_uniform/min*$
_class
loc:@pi/dense_1/kernel*
T0*
_output_shapes
: 
?
0pi/dense_1/kernel/Initializer/random_uniform/mulMul:pi/dense_1/kernel/Initializer/random_uniform/RandomUniform0pi/dense_1/kernel/Initializer/random_uniform/sub* 
_output_shapes
:
??*
T0*$
_class
loc:@pi/dense_1/kernel
?
,pi/dense_1/kernel/Initializer/random_uniformAdd0pi/dense_1/kernel/Initializer/random_uniform/mul0pi/dense_1/kernel/Initializer/random_uniform/min*$
_class
loc:@pi/dense_1/kernel*
T0* 
_output_shapes
:
??
?
pi/dense_1/kernel
VariableV2*
dtype0*$
_class
loc:@pi/dense_1/kernel*
shape:
??*
shared_name * 
_output_shapes
:
??*
	container 
?
pi/dense_1/kernel/AssignAssignpi/dense_1/kernel,pi/dense_1/kernel/Initializer/random_uniform* 
_output_shapes
:
??*
validate_shape(*
T0*
use_locking(*$
_class
loc:@pi/dense_1/kernel
?
pi/dense_1/kernel/readIdentitypi/dense_1/kernel* 
_output_shapes
:
??*
T0*$
_class
loc:@pi/dense_1/kernel
?
!pi/dense_1/bias/Initializer/zerosConst*"
_class
loc:@pi/dense_1/bias*
dtype0*
valueB?*    *
_output_shapes	
:?
?
pi/dense_1/bias
VariableV2*
shape:?*"
_class
loc:@pi/dense_1/bias*
_output_shapes	
:?*
	container *
dtype0*
shared_name 
?
pi/dense_1/bias/AssignAssignpi/dense_1/bias!pi/dense_1/bias/Initializer/zeros*
use_locking(*
T0*
validate_shape(*
_output_shapes	
:?*"
_class
loc:@pi/dense_1/bias
{
pi/dense_1/bias/readIdentitypi/dense_1/bias*
_output_shapes	
:?*
T0*"
_class
loc:@pi/dense_1/bias
?
pi/dense_1/MatMulMatMulpi/dense/Tanhpi/dense_1/kernel/read*
T0*
transpose_b( *(
_output_shapes
:??????????*
transpose_a( 
?
pi/dense_1/BiasAddBiasAddpi/dense_1/MatMulpi/dense_1/bias/read*
data_formatNHWC*
T0*(
_output_shapes
:??????????
^
pi/dense_1/TanhTanhpi/dense_1/BiasAdd*(
_output_shapes
:??????????*
T0
?
2pi/dense_2/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*
valueB"      *$
_class
loc:@pi/dense_2/kernel*
dtype0
?
0pi/dense_2/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *$
_class
loc:@pi/dense_2/kernel*
dtype0*
valueB
 *?(?
?
0pi/dense_2/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *?(>*$
_class
loc:@pi/dense_2/kernel
?
:pi/dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform2pi/dense_2/kernel/Initializer/random_uniform/shape*
seed2.*
T0*$
_class
loc:@pi/dense_2/kernel*
_output_shapes
:	?*
dtype0*

seed 
?
0pi/dense_2/kernel/Initializer/random_uniform/subSub0pi/dense_2/kernel/Initializer/random_uniform/max0pi/dense_2/kernel/Initializer/random_uniform/min*$
_class
loc:@pi/dense_2/kernel*
T0*
_output_shapes
: 
?
0pi/dense_2/kernel/Initializer/random_uniform/mulMul:pi/dense_2/kernel/Initializer/random_uniform/RandomUniform0pi/dense_2/kernel/Initializer/random_uniform/sub*
T0*
_output_shapes
:	?*$
_class
loc:@pi/dense_2/kernel
?
,pi/dense_2/kernel/Initializer/random_uniformAdd0pi/dense_2/kernel/Initializer/random_uniform/mul0pi/dense_2/kernel/Initializer/random_uniform/min*$
_class
loc:@pi/dense_2/kernel*
_output_shapes
:	?*
T0
?
pi/dense_2/kernel
VariableV2*
shape:	?*
	container *
shared_name *$
_class
loc:@pi/dense_2/kernel*
_output_shapes
:	?*
dtype0
?
pi/dense_2/kernel/AssignAssignpi/dense_2/kernel,pi/dense_2/kernel/Initializer/random_uniform*
use_locking(*
validate_shape(*
T0*$
_class
loc:@pi/dense_2/kernel*
_output_shapes
:	?
?
pi/dense_2/kernel/readIdentitypi/dense_2/kernel*
_output_shapes
:	?*
T0*$
_class
loc:@pi/dense_2/kernel
?
!pi/dense_2/bias/Initializer/zerosConst*
valueB*    *
dtype0*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:
?
pi/dense_2/bias
VariableV2*
dtype0*
shape:*
	container *
shared_name *
_output_shapes
:*"
_class
loc:@pi/dense_2/bias
?
pi/dense_2/bias/AssignAssignpi/dense_2/bias!pi/dense_2/bias/Initializer/zeros*
use_locking(*
_output_shapes
:*
T0*
validate_shape(*"
_class
loc:@pi/dense_2/bias
z
pi/dense_2/bias/readIdentitypi/dense_2/bias*
_output_shapes
:*
T0*"
_class
loc:@pi/dense_2/bias
?
pi/dense_2/MatMulMatMulpi/dense_1/Tanhpi/dense_2/kernel/read*
T0*
transpose_b( *'
_output_shapes
:?????????*
transpose_a( 
?
pi/dense_2/BiasAddBiasAddpi/dense_2/MatMulpi/dense_2/bias/read*
data_formatNHWC*
T0*'
_output_shapes
:?????????
i
pi/log_std/initial_valueConst*
_output_shapes
:*
valueB"   ?   ?*
dtype0
v

pi/log_std
VariableV2*
	container *
dtype0*
shared_name *
_output_shapes
:*
shape:
?
pi/log_std/AssignAssign
pi/log_stdpi/log_std/initial_value*
T0*
validate_shape(*
use_locking(*
_class
loc:@pi/log_std*
_output_shapes
:
k
pi/log_std/readIdentity
pi/log_std*
T0*
_class
loc:@pi/log_std*
_output_shapes
:
C
pi/ExpExppi/log_std/read*
T0*
_output_shapes
:
Z
pi/ShapeShapepi/dense_2/BiasAdd*
_output_shapes
:*
out_type0*
T0
Z
pi/random_normal/meanConst*
valueB
 *    *
_output_shapes
: *
dtype0
\
pi/random_normal/stddevConst*
valueB
 *  ??*
_output_shapes
: *
dtype0
?
%pi/random_normal/RandomStandardNormalRandomStandardNormalpi/Shape*
seed2C*
dtype0*
T0*

seed *'
_output_shapes
:?????????
?
pi/random_normal/mulMul%pi/random_normal/RandomStandardNormalpi/random_normal/stddev*
T0*'
_output_shapes
:?????????
v
pi/random_normalAddpi/random_normal/mulpi/random_normal/mean*'
_output_shapes
:?????????*
T0
Y
pi/mulMulpi/random_normalpi/Exp*
T0*'
_output_shapes
:?????????
[
pi/addAddpi/dense_2/BiasAddpi/mul*
T0*'
_output_shapes
:?????????
b
pi/subSubPlaceholder_1pi/dense_2/BiasAdd*'
_output_shapes
:?????????*
T0
E
pi/Exp_1Exppi/log_std/read*
T0*
_output_shapes
:
O

pi/add_1/yConst*
valueB
 *w?+2*
_output_shapes
: *
dtype0
J
pi/add_1Addpi/Exp_1
pi/add_1/y*
T0*
_output_shapes
:
Y

pi/truedivRealDivpi/subpi/add_1*
T0*'
_output_shapes
:?????????
M
pi/pow/yConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
U
pi/powPow
pi/truedivpi/pow/y*'
_output_shapes
:?????????*
T0
O

pi/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @
Q
pi/mul_1Mul
pi/mul_1/xpi/log_std/read*
_output_shapes
:*
T0
S
pi/add_2Addpi/powpi/mul_1*'
_output_shapes
:?????????*
T0
O

pi/add_3/yConst*
valueB
 *????*
dtype0*
_output_shapes
: 
W
pi/add_3Addpi/add_2
pi/add_3/y*'
_output_shapes
:?????????*
T0
O

pi/mul_2/xConst*
_output_shapes
: *
valueB
 *   ?*
dtype0
W
pi/mul_2Mul
pi/mul_2/xpi/add_3*
T0*'
_output_shapes
:?????????
Z
pi/Sum/reduction_indicesConst*
dtype0*
value	B :*
_output_shapes
: 
|
pi/SumSumpi/mul_2pi/Sum/reduction_indices*#
_output_shapes
:?????????*

Tidx0*
T0*
	keep_dims( 
]
pi/sub_1Subpi/addpi/dense_2/BiasAdd*
T0*'
_output_shapes
:?????????
E
pi/Exp_2Exppi/log_std/read*
_output_shapes
:*
T0
O

pi/add_4/yConst*
_output_shapes
: *
valueB
 *w?+2*
dtype0
J
pi/add_4Addpi/Exp_2
pi/add_4/y*
T0*
_output_shapes
:
]
pi/truediv_1RealDivpi/sub_1pi/add_4*'
_output_shapes
:?????????*
T0
O

pi/pow_1/yConst*
dtype0*
valueB
 *   @*
_output_shapes
: 
[
pi/pow_1Powpi/truediv_1
pi/pow_1/y*
T0*'
_output_shapes
:?????????
O

pi/mul_3/xConst*
_output_shapes
: *
dtype0*
valueB
 *   @
Q
pi/mul_3Mul
pi/mul_3/xpi/log_std/read*
T0*
_output_shapes
:
U
pi/add_5Addpi/pow_1pi/mul_3*
T0*'
_output_shapes
:?????????
O

pi/add_6/yConst*
dtype0*
valueB
 *????*
_output_shapes
: 
W
pi/add_6Addpi/add_5
pi/add_6/y*
T0*'
_output_shapes
:?????????
O

pi/mul_4/xConst*
valueB
 *   ?*
_output_shapes
: *
dtype0
W
pi/mul_4Mul
pi/mul_4/xpi/add_6*'
_output_shapes
:?????????*
T0
\
pi/Sum_1/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :
?
pi/Sum_1Sumpi/mul_4pi/Sum_1/reduction_indices*
T0*#
_output_shapes
:?????????*

Tidx0*
	keep_dims( 
q
pi/PlaceholderPlaceholder*'
_output_shapes
:?????????*
dtype0*
shape:?????????
s
pi/Placeholder_1Placeholder*'
_output_shapes
:?????????*
shape:?????????*
dtype0
O

pi/mul_5/xConst*
valueB
 *   @*
_output_shapes
: *
dtype0
Q
pi/mul_5Mul
pi/mul_5/xpi/log_std/read*
_output_shapes
:*
T0
>
pi/Exp_3Exppi/mul_5*
_output_shapes
:*
T0
O

pi/mul_6/xConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
_
pi/mul_6Mul
pi/mul_6/xpi/Placeholder_1*'
_output_shapes
:?????????*
T0
K
pi/Exp_4Exppi/mul_6*
T0*'
_output_shapes
:?????????
e
pi/sub_2Subpi/Placeholderpi/dense_2/BiasAdd*
T0*'
_output_shapes
:?????????
O

pi/pow_2/yConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
W
pi/pow_2Powpi/sub_2
pi/pow_2/y*
T0*'
_output_shapes
:?????????
U
pi/add_7Addpi/pow_2pi/Exp_3*
T0*'
_output_shapes
:?????????
O

pi/add_8/yConst*
_output_shapes
: *
dtype0*
valueB
 *w?+2
W
pi/add_8Addpi/Exp_4
pi/add_8/y*
T0*'
_output_shapes
:?????????
]
pi/truediv_2RealDivpi/add_7pi/add_8*
T0*'
_output_shapes
:?????????
O

pi/sub_3/yConst*
dtype0*
valueB
 *  ??*
_output_shapes
: 
[
pi/sub_3Subpi/truediv_2
pi/sub_3/y*'
_output_shapes
:?????????*
T0
O

pi/mul_7/xConst*
_output_shapes
: *
valueB
 *   ?*
dtype0
W
pi/mul_7Mul
pi/mul_7/xpi/sub_3*'
_output_shapes
:?????????*
T0
]
pi/add_9Addpi/mul_7pi/Placeholder_1*
T0*'
_output_shapes
:?????????
\
pi/sub_4Subpi/add_9pi/log_std/read*
T0*'
_output_shapes
:?????????
\
pi/Sum_2/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :
?
pi/Sum_2Sumpi/sub_4pi/Sum_2/reduction_indices*

Tidx0*
T0*#
_output_shapes
:?????????*
	keep_dims( 
R
pi/ConstConst*
valueB: *
_output_shapes
:*
dtype0
a
pi/MeanMeanpi/Sum_2pi/Const*
	keep_dims( *
_output_shapes
: *
T0*

Tidx0
P
pi/add_10/yConst*
_output_shapes
: *
dtype0*
valueB
 *ǟ??
S
	pi/add_10Addpi/log_std/readpi/add_10/y*
T0*
_output_shapes
:
e
pi/Sum_3/reduction_indicesConst*
valueB :
?????????*
dtype0*
_output_shapes
: 
t
pi/Sum_3Sum	pi/add_10pi/Sum_3/reduction_indices*
	keep_dims( *
_output_shapes
: *

Tidx0*
T0
M

pi/Const_1Const*
dtype0*
valueB *
_output_shapes
: 
e
	pi/Mean_1Meanpi/Sum_3
pi/Const_1*
T0*
	keep_dims( *
_output_shapes
: *

Tidx0
?
0vf/dense/kernel/Initializer/random_uniform/shapeConst*"
_class
loc:@vf/dense/kernel*
valueB"<      *
dtype0*
_output_shapes
:
?
.vf/dense/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *?*"
_class
loc:@vf/dense/kernel
?
.vf/dense/kernel/Initializer/random_uniform/maxConst*
valueB
 *>*
dtype0*
_output_shapes
: *"
_class
loc:@vf/dense/kernel
?
8vf/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform0vf/dense/kernel/Initializer/random_uniform/shape*
T0*
_output_shapes
:	<?*
dtype0*

seed *"
_class
loc:@vf/dense/kernel*
seed2?
?
.vf/dense/kernel/Initializer/random_uniform/subSub.vf/dense/kernel/Initializer/random_uniform/max.vf/dense/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*"
_class
loc:@vf/dense/kernel
?
.vf/dense/kernel/Initializer/random_uniform/mulMul8vf/dense/kernel/Initializer/random_uniform/RandomUniform.vf/dense/kernel/Initializer/random_uniform/sub*"
_class
loc:@vf/dense/kernel*
T0*
_output_shapes
:	<?
?
*vf/dense/kernel/Initializer/random_uniformAdd.vf/dense/kernel/Initializer/random_uniform/mul.vf/dense/kernel/Initializer/random_uniform/min*
_output_shapes
:	<?*
T0*"
_class
loc:@vf/dense/kernel
?
vf/dense/kernel
VariableV2*
shape:	<?*
	container *
_output_shapes
:	<?*
shared_name *"
_class
loc:@vf/dense/kernel*
dtype0
?
vf/dense/kernel/AssignAssignvf/dense/kernel*vf/dense/kernel/Initializer/random_uniform*
validate_shape(*
use_locking(*
_output_shapes
:	<?*"
_class
loc:@vf/dense/kernel*
T0

vf/dense/kernel/readIdentityvf/dense/kernel*
T0*"
_class
loc:@vf/dense/kernel*
_output_shapes
:	<?
?
vf/dense/bias/Initializer/zerosConst* 
_class
loc:@vf/dense/bias*
dtype0*
valueB?*    *
_output_shapes	
:?
?
vf/dense/bias
VariableV2*
dtype0* 
_class
loc:@vf/dense/bias*
shape:?*
shared_name *
	container *
_output_shapes	
:?
?
vf/dense/bias/AssignAssignvf/dense/biasvf/dense/bias/Initializer/zeros*
use_locking(*
_output_shapes	
:?*
T0*
validate_shape(* 
_class
loc:@vf/dense/bias
u
vf/dense/bias/readIdentityvf/dense/bias*
T0*
_output_shapes	
:?* 
_class
loc:@vf/dense/bias
?
vf/dense/MatMulMatMulPlaceholdervf/dense/kernel/read*
T0*
transpose_b( *(
_output_shapes
:??????????*
transpose_a( 
?
vf/dense/BiasAddBiasAddvf/dense/MatMulvf/dense/bias/read*(
_output_shapes
:??????????*
data_formatNHWC*
T0
Z
vf/dense/TanhTanhvf/dense/BiasAdd*
T0*(
_output_shapes
:??????????
?
2vf/dense_1/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*
valueB"      *
dtype0*$
_class
loc:@vf/dense_1/kernel
?
0vf/dense_1/kernel/Initializer/random_uniform/minConst*$
_class
loc:@vf/dense_1/kernel*
valueB
 *׳ݽ*
dtype0*
_output_shapes
: 
?
0vf/dense_1/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *$
_class
loc:@vf/dense_1/kernel*
valueB
 *׳?=*
dtype0
?
:vf/dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform2vf/dense_1/kernel/Initializer/random_uniform/shape*

seed *
T0* 
_output_shapes
:
??*
dtype0*$
_class
loc:@vf/dense_1/kernel*
seed2?
?
0vf/dense_1/kernel/Initializer/random_uniform/subSub0vf/dense_1/kernel/Initializer/random_uniform/max0vf/dense_1/kernel/Initializer/random_uniform/min*$
_class
loc:@vf/dense_1/kernel*
_output_shapes
: *
T0
?
0vf/dense_1/kernel/Initializer/random_uniform/mulMul:vf/dense_1/kernel/Initializer/random_uniform/RandomUniform0vf/dense_1/kernel/Initializer/random_uniform/sub*
T0* 
_output_shapes
:
??*$
_class
loc:@vf/dense_1/kernel
?
,vf/dense_1/kernel/Initializer/random_uniformAdd0vf/dense_1/kernel/Initializer/random_uniform/mul0vf/dense_1/kernel/Initializer/random_uniform/min* 
_output_shapes
:
??*$
_class
loc:@vf/dense_1/kernel*
T0
?
vf/dense_1/kernel
VariableV2*
shared_name * 
_output_shapes
:
??*
shape:
??*
dtype0*$
_class
loc:@vf/dense_1/kernel*
	container 
?
vf/dense_1/kernel/AssignAssignvf/dense_1/kernel,vf/dense_1/kernel/Initializer/random_uniform*
T0*
validate_shape(*$
_class
loc:@vf/dense_1/kernel* 
_output_shapes
:
??*
use_locking(
?
vf/dense_1/kernel/readIdentityvf/dense_1/kernel*
T0* 
_output_shapes
:
??*$
_class
loc:@vf/dense_1/kernel
?
!vf/dense_1/bias/Initializer/zerosConst*
_output_shapes	
:?*
dtype0*
valueB?*    *"
_class
loc:@vf/dense_1/bias
?
vf/dense_1/bias
VariableV2*
shared_name *"
_class
loc:@vf/dense_1/bias*
shape:?*
dtype0*
	container *
_output_shapes	
:?
?
vf/dense_1/bias/AssignAssignvf/dense_1/bias!vf/dense_1/bias/Initializer/zeros*
_output_shapes	
:?*
T0*
use_locking(*"
_class
loc:@vf/dense_1/bias*
validate_shape(
{
vf/dense_1/bias/readIdentityvf/dense_1/bias*
T0*"
_class
loc:@vf/dense_1/bias*
_output_shapes	
:?
?
vf/dense_1/MatMulMatMulvf/dense/Tanhvf/dense_1/kernel/read*
transpose_a( *
T0*(
_output_shapes
:??????????*
transpose_b( 
?
vf/dense_1/BiasAddBiasAddvf/dense_1/MatMulvf/dense_1/bias/read*
data_formatNHWC*
T0*(
_output_shapes
:??????????
^
vf/dense_1/TanhTanhvf/dense_1/BiasAdd*(
_output_shapes
:??????????*
T0
?
2vf/dense_2/kernel/Initializer/random_uniform/shapeConst*
valueB"      *$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:*
dtype0
?
0vf/dense_2/kernel/Initializer/random_uniform/minConst*
dtype0*$
_class
loc:@vf/dense_2/kernel*
valueB
 *Iv?*
_output_shapes
: 
?
0vf/dense_2/kernel/Initializer/random_uniform/maxConst*
valueB
 *Iv>*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
: *
dtype0
?
:vf/dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform2vf/dense_2/kernel/Initializer/random_uniform/shape*
_output_shapes
:	?*$
_class
loc:@vf/dense_2/kernel*

seed *
dtype0*
T0*
seed2?
?
0vf/dense_2/kernel/Initializer/random_uniform/subSub0vf/dense_2/kernel/Initializer/random_uniform/max0vf/dense_2/kernel/Initializer/random_uniform/min*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
: *
T0
?
0vf/dense_2/kernel/Initializer/random_uniform/mulMul:vf/dense_2/kernel/Initializer/random_uniform/RandomUniform0vf/dense_2/kernel/Initializer/random_uniform/sub*
T0*
_output_shapes
:	?*$
_class
loc:@vf/dense_2/kernel
?
,vf/dense_2/kernel/Initializer/random_uniformAdd0vf/dense_2/kernel/Initializer/random_uniform/mul0vf/dense_2/kernel/Initializer/random_uniform/min*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	?*
T0
?
vf/dense_2/kernel
VariableV2*$
_class
loc:@vf/dense_2/kernel*
shared_name *
dtype0*
shape:	?*
	container *
_output_shapes
:	?
?
vf/dense_2/kernel/AssignAssignvf/dense_2/kernel,vf/dense_2/kernel/Initializer/random_uniform*
use_locking(*
validate_shape(*
T0*
_output_shapes
:	?*$
_class
loc:@vf/dense_2/kernel
?
vf/dense_2/kernel/readIdentityvf/dense_2/kernel*
T0*
_output_shapes
:	?*$
_class
loc:@vf/dense_2/kernel
?
!vf/dense_2/bias/Initializer/zerosConst*
valueB*    *"
_class
loc:@vf/dense_2/bias*
dtype0*
_output_shapes
:
?
vf/dense_2/bias
VariableV2*"
_class
loc:@vf/dense_2/bias*
	container *
_output_shapes
:*
dtype0*
shared_name *
shape:
?
vf/dense_2/bias/AssignAssignvf/dense_2/bias!vf/dense_2/bias/Initializer/zeros*
T0*
use_locking(*"
_class
loc:@vf/dense_2/bias*
_output_shapes
:*
validate_shape(
z
vf/dense_2/bias/readIdentityvf/dense_2/bias*
_output_shapes
:*"
_class
loc:@vf/dense_2/bias*
T0
?
vf/dense_2/MatMulMatMulvf/dense_1/Tanhvf/dense_2/kernel/read*
transpose_a( *
T0*
transpose_b( *'
_output_shapes
:?????????
?
vf/dense_2/BiasAddBiasAddvf/dense_2/MatMulvf/dense_2/bias/read*
T0*'
_output_shapes
:?????????*
data_formatNHWC
n

vf/SqueezeSqueezevf/dense_2/BiasAdd*#
_output_shapes
:?????????*
T0*
squeeze_dims

?
0vc/dense/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*
valueB"<      *
dtype0*"
_class
loc:@vc/dense/kernel
?
.vc/dense/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *"
_class
loc:@vc/dense/kernel*
valueB
 *?*
dtype0
?
.vc/dense/kernel/Initializer/random_uniform/maxConst*
valueB
 *>*"
_class
loc:@vc/dense/kernel*
_output_shapes
: *
dtype0
?
8vc/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform0vc/dense/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes
:	<?*
seed2?*"
_class
loc:@vc/dense/kernel*

seed *
T0
?
.vc/dense/kernel/Initializer/random_uniform/subSub.vc/dense/kernel/Initializer/random_uniform/max.vc/dense/kernel/Initializer/random_uniform/min*
T0*"
_class
loc:@vc/dense/kernel*
_output_shapes
: 
?
.vc/dense/kernel/Initializer/random_uniform/mulMul8vc/dense/kernel/Initializer/random_uniform/RandomUniform.vc/dense/kernel/Initializer/random_uniform/sub*
T0*"
_class
loc:@vc/dense/kernel*
_output_shapes
:	<?
?
*vc/dense/kernel/Initializer/random_uniformAdd.vc/dense/kernel/Initializer/random_uniform/mul.vc/dense/kernel/Initializer/random_uniform/min*"
_class
loc:@vc/dense/kernel*
_output_shapes
:	<?*
T0
?
vc/dense/kernel
VariableV2*
shared_name *
	container *"
_class
loc:@vc/dense/kernel*
dtype0*
_output_shapes
:	<?*
shape:	<?
?
vc/dense/kernel/AssignAssignvc/dense/kernel*vc/dense/kernel/Initializer/random_uniform*
use_locking(*"
_class
loc:@vc/dense/kernel*
_output_shapes
:	<?*
T0*
validate_shape(

vc/dense/kernel/readIdentityvc/dense/kernel*
_output_shapes
:	<?*"
_class
loc:@vc/dense/kernel*
T0
?
vc/dense/bias/Initializer/zerosConst*
valueB?*    *
_output_shapes	
:?* 
_class
loc:@vc/dense/bias*
dtype0
?
vc/dense/bias
VariableV2*
dtype0*
shared_name *
shape:?*
_output_shapes	
:?* 
_class
loc:@vc/dense/bias*
	container 
?
vc/dense/bias/AssignAssignvc/dense/biasvc/dense/bias/Initializer/zeros*
validate_shape(* 
_class
loc:@vc/dense/bias*
use_locking(*
T0*
_output_shapes	
:?
u
vc/dense/bias/readIdentityvc/dense/bias*
_output_shapes	
:?*
T0* 
_class
loc:@vc/dense/bias
?
vc/dense/MatMulMatMulPlaceholdervc/dense/kernel/read*
transpose_a( *
transpose_b( *
T0*(
_output_shapes
:??????????
?
vc/dense/BiasAddBiasAddvc/dense/MatMulvc/dense/bias/read*
data_formatNHWC*(
_output_shapes
:??????????*
T0
Z
vc/dense/TanhTanhvc/dense/BiasAdd*(
_output_shapes
:??????????*
T0
?
2vc/dense_1/kernel/Initializer/random_uniform/shapeConst*
dtype0*
valueB"      *
_output_shapes
:*$
_class
loc:@vc/dense_1/kernel
?
0vc/dense_1/kernel/Initializer/random_uniform/minConst*$
_class
loc:@vc/dense_1/kernel*
valueB
 *׳ݽ*
dtype0*
_output_shapes
: 
?
0vc/dense_1/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *
valueB
 *׳?=*$
_class
loc:@vc/dense_1/kernel*
dtype0
?
:vc/dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform2vc/dense_1/kernel/Initializer/random_uniform/shape*
seed2?* 
_output_shapes
:
??*

seed *$
_class
loc:@vc/dense_1/kernel*
T0*
dtype0
?
0vc/dense_1/kernel/Initializer/random_uniform/subSub0vc/dense_1/kernel/Initializer/random_uniform/max0vc/dense_1/kernel/Initializer/random_uniform/min*
T0*$
_class
loc:@vc/dense_1/kernel*
_output_shapes
: 
?
0vc/dense_1/kernel/Initializer/random_uniform/mulMul:vc/dense_1/kernel/Initializer/random_uniform/RandomUniform0vc/dense_1/kernel/Initializer/random_uniform/sub*$
_class
loc:@vc/dense_1/kernel*
T0* 
_output_shapes
:
??
?
,vc/dense_1/kernel/Initializer/random_uniformAdd0vc/dense_1/kernel/Initializer/random_uniform/mul0vc/dense_1/kernel/Initializer/random_uniform/min*$
_class
loc:@vc/dense_1/kernel* 
_output_shapes
:
??*
T0
?
vc/dense_1/kernel
VariableV2*
shape:
??* 
_output_shapes
:
??*$
_class
loc:@vc/dense_1/kernel*
shared_name *
	container *
dtype0
?
vc/dense_1/kernel/AssignAssignvc/dense_1/kernel,vc/dense_1/kernel/Initializer/random_uniform*
use_locking(*$
_class
loc:@vc/dense_1/kernel* 
_output_shapes
:
??*
validate_shape(*
T0
?
vc/dense_1/kernel/readIdentityvc/dense_1/kernel* 
_output_shapes
:
??*$
_class
loc:@vc/dense_1/kernel*
T0
?
!vc/dense_1/bias/Initializer/zerosConst*
dtype0*"
_class
loc:@vc/dense_1/bias*
_output_shapes	
:?*
valueB?*    
?
vc/dense_1/bias
VariableV2*
_output_shapes	
:?*
	container *
dtype0*
shared_name *
shape:?*"
_class
loc:@vc/dense_1/bias
?
vc/dense_1/bias/AssignAssignvc/dense_1/bias!vc/dense_1/bias/Initializer/zeros*"
_class
loc:@vc/dense_1/bias*
T0*
use_locking(*
validate_shape(*
_output_shapes	
:?
{
vc/dense_1/bias/readIdentityvc/dense_1/bias*"
_class
loc:@vc/dense_1/bias*
_output_shapes	
:?*
T0
?
vc/dense_1/MatMulMatMulvc/dense/Tanhvc/dense_1/kernel/read*
T0*
transpose_a( *(
_output_shapes
:??????????*
transpose_b( 
?
vc/dense_1/BiasAddBiasAddvc/dense_1/MatMulvc/dense_1/bias/read*(
_output_shapes
:??????????*
data_formatNHWC*
T0
^
vc/dense_1/TanhTanhvc/dense_1/BiasAdd*(
_output_shapes
:??????????*
T0
?
2vc/dense_2/kernel/Initializer/random_uniform/shapeConst*
valueB"      *$
_class
loc:@vc/dense_2/kernel*
dtype0*
_output_shapes
:
?
0vc/dense_2/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *
dtype0*
valueB
 *Iv?*$
_class
loc:@vc/dense_2/kernel
?
0vc/dense_2/kernel/Initializer/random_uniform/maxConst*
valueB
 *Iv>*
dtype0*
_output_shapes
: *$
_class
loc:@vc/dense_2/kernel
?
:vc/dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform2vc/dense_2/kernel/Initializer/random_uniform/shape*
dtype0*
T0*
_output_shapes
:	?*$
_class
loc:@vc/dense_2/kernel*
seed2?*

seed 
?
0vc/dense_2/kernel/Initializer/random_uniform/subSub0vc/dense_2/kernel/Initializer/random_uniform/max0vc/dense_2/kernel/Initializer/random_uniform/min*$
_class
loc:@vc/dense_2/kernel*
T0*
_output_shapes
: 
?
0vc/dense_2/kernel/Initializer/random_uniform/mulMul:vc/dense_2/kernel/Initializer/random_uniform/RandomUniform0vc/dense_2/kernel/Initializer/random_uniform/sub*
_output_shapes
:	?*$
_class
loc:@vc/dense_2/kernel*
T0
?
,vc/dense_2/kernel/Initializer/random_uniformAdd0vc/dense_2/kernel/Initializer/random_uniform/mul0vc/dense_2/kernel/Initializer/random_uniform/min*
T0*$
_class
loc:@vc/dense_2/kernel*
_output_shapes
:	?
?
vc/dense_2/kernel
VariableV2*
	container *$
_class
loc:@vc/dense_2/kernel*
_output_shapes
:	?*
shape:	?*
shared_name *
dtype0
?
vc/dense_2/kernel/AssignAssignvc/dense_2/kernel,vc/dense_2/kernel/Initializer/random_uniform*
use_locking(*
T0*
validate_shape(*$
_class
loc:@vc/dense_2/kernel*
_output_shapes
:	?
?
vc/dense_2/kernel/readIdentityvc/dense_2/kernel*
_output_shapes
:	?*
T0*$
_class
loc:@vc/dense_2/kernel
?
!vc/dense_2/bias/Initializer/zerosConst*
valueB*    *"
_class
loc:@vc/dense_2/bias*
_output_shapes
:*
dtype0
?
vc/dense_2/bias
VariableV2*
shape:*
shared_name *
_output_shapes
:*"
_class
loc:@vc/dense_2/bias*
dtype0*
	container 
?
vc/dense_2/bias/AssignAssignvc/dense_2/bias!vc/dense_2/bias/Initializer/zeros*
validate_shape(*"
_class
loc:@vc/dense_2/bias*
_output_shapes
:*
use_locking(*
T0
z
vc/dense_2/bias/readIdentityvc/dense_2/bias*"
_class
loc:@vc/dense_2/bias*
T0*
_output_shapes
:
?
vc/dense_2/MatMulMatMulvc/dense_1/Tanhvc/dense_2/kernel/read*
transpose_a( *
transpose_b( *'
_output_shapes
:?????????*
T0
?
vc/dense_2/BiasAddBiasAddvc/dense_2/MatMulvc/dense_2/bias/read*
T0*'
_output_shapes
:?????????*
data_formatNHWC
n

vc/SqueezeSqueezevc/dense_2/BiasAdd*#
_output_shapes
:?????????*
T0*
squeeze_dims

@
NegNegpi/Sum*
T0*#
_output_shapes
:?????????
O
ConstConst*
_output_shapes
:*
valueB: *
dtype0
V
MeanMeanNegConst*
	keep_dims( *
_output_shapes
: *
T0*

Tidx0
O
subSubpi/SumPlaceholder_6*
T0*#
_output_shapes
:?????????
=
ExpExpsub*#
_output_shapes
:?????????*
T0
L
mulMulExpPlaceholder_2*
T0*#
_output_shapes
:?????????
Q
Const_1Const*
_output_shapes
:*
dtype0*
valueB: 
Z
Mean_1MeanmulConst_1*
	keep_dims( *
T0*
_output_shapes
: *

Tidx0
N
mul_1MulExpPlaceholder_3*#
_output_shapes
:?????????*
T0
Q
Const_2Const*
valueB: *
_output_shapes
:*
dtype0
\
Mean_2Meanmul_1Const_2*

Tidx0*
	keep_dims( *
_output_shapes
: *
T0
L
mul_2/xConst*
dtype0*
_output_shapes
: *
valueB
 *    
A
mul_2Mulmul_2/x	pi/Mean_1*
T0*
_output_shapes
: 
:
addAddMean_1mul_2*
T0*
_output_shapes
: 
2
Neg_1Negadd*
_output_shapes
: *
T0
R
gradients/ShapeConst*
valueB *
_output_shapes
: *
dtype0
X
gradients/grad_ys_0Const*
_output_shapes
: *
dtype0*
valueB
 *  ??
o
gradients/FillFillgradients/Shapegradients/grad_ys_0*
T0*
_output_shapes
: *

index_type0
P
gradients/Neg_1_grad/NegNeggradients/Fill*
T0*
_output_shapes
: 
m
#gradients/Mean_1_grad/Reshape/shapeConst*
valueB:*
_output_shapes
:*
dtype0
?
gradients/Mean_1_grad/ReshapeReshapegradients/Neg_1_grad/Neg#gradients/Mean_1_grad/Reshape/shape*
_output_shapes
:*
T0*
Tshape0
^
gradients/Mean_1_grad/ShapeShapemul*
T0*
_output_shapes
:*
out_type0
?
gradients/Mean_1_grad/TileTilegradients/Mean_1_grad/Reshapegradients/Mean_1_grad/Shape*

Tmultiples0*#
_output_shapes
:?????????*
T0
`
gradients/Mean_1_grad/Shape_1Shapemul*
out_type0*
_output_shapes
:*
T0
`
gradients/Mean_1_grad/Shape_2Const*
_output_shapes
: *
dtype0*
valueB 
e
gradients/Mean_1_grad/ConstConst*
valueB: *
_output_shapes
:*
dtype0
?
gradients/Mean_1_grad/ProdProdgradients/Mean_1_grad/Shape_1gradients/Mean_1_grad/Const*

Tidx0*
T0*
_output_shapes
: *
	keep_dims( 
g
gradients/Mean_1_grad/Const_1Const*
dtype0*
valueB: *
_output_shapes
:
?
gradients/Mean_1_grad/Prod_1Prodgradients/Mean_1_grad/Shape_2gradients/Mean_1_grad/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
a
gradients/Mean_1_grad/Maximum/yConst*
_output_shapes
: *
dtype0*
value	B :
?
gradients/Mean_1_grad/MaximumMaximumgradients/Mean_1_grad/Prod_1gradients/Mean_1_grad/Maximum/y*
_output_shapes
: *
T0
?
gradients/Mean_1_grad/floordivFloorDivgradients/Mean_1_grad/Prodgradients/Mean_1_grad/Maximum*
_output_shapes
: *
T0
?
gradients/Mean_1_grad/CastCastgradients/Mean_1_grad/floordiv*

SrcT0*
_output_shapes
: *
Truncate( *

DstT0
?
gradients/Mean_1_grad/truedivRealDivgradients/Mean_1_grad/Tilegradients/Mean_1_grad/Cast*#
_output_shapes
:?????????*
T0
e
gradients/mul_2_grad/MulMulgradients/Neg_1_grad/Neg	pi/Mean_1*
T0*
_output_shapes
: 
e
gradients/mul_2_grad/Mul_1Mulgradients/Neg_1_grad/Negmul_2/x*
_output_shapes
: *
T0
[
gradients/mul_grad/ShapeShapeExp*
out_type0*
T0*
_output_shapes
:
g
gradients/mul_grad/Shape_1ShapePlaceholder_2*
_output_shapes
:*
out_type0*
T0
?
(gradients/mul_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/mul_grad/Shapegradients/mul_grad/Shape_1*2
_output_shapes 
:?????????:?????????*
T0
y
gradients/mul_grad/MulMulgradients/Mean_1_grad/truedivPlaceholder_2*#
_output_shapes
:?????????*
T0
?
gradients/mul_grad/SumSumgradients/mul_grad/Mul(gradients/mul_grad/BroadcastGradientArgs*

Tidx0*
_output_shapes
:*
	keep_dims( *
T0
?
gradients/mul_grad/ReshapeReshapegradients/mul_grad/Sumgradients/mul_grad/Shape*
T0*#
_output_shapes
:?????????*
Tshape0
q
gradients/mul_grad/Mul_1MulExpgradients/Mean_1_grad/truediv*#
_output_shapes
:?????????*
T0
?
gradients/mul_grad/Sum_1Sumgradients/mul_grad/Mul_1*gradients/mul_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*

Tidx0*
_output_shapes
:
?
gradients/mul_grad/Reshape_1Reshapegradients/mul_grad/Sum_1gradients/mul_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:?????????
i
&gradients/pi/Mean_1_grad/Reshape/shapeConst*
valueB *
dtype0*
_output_shapes
: 
?
 gradients/pi/Mean_1_grad/ReshapeReshapegradients/mul_2_grad/Mul_1&gradients/pi/Mean_1_grad/Reshape/shape*
Tshape0*
_output_shapes
: *
T0
a
gradients/pi/Mean_1_grad/ConstConst*
dtype0*
_output_shapes
: *
valueB 
?
gradients/pi/Mean_1_grad/TileTile gradients/pi/Mean_1_grad/Reshapegradients/pi/Mean_1_grad/Const*
_output_shapes
: *

Tmultiples0*
T0
e
 gradients/pi/Mean_1_grad/Const_1Const*
_output_shapes
: *
valueB
 *  ??*
dtype0
?
 gradients/pi/Mean_1_grad/truedivRealDivgradients/pi/Mean_1_grad/Tile gradients/pi/Mean_1_grad/Const_1*
_output_shapes
: *
T0
l
gradients/Exp_grad/mulMulgradients/mul_grad/ReshapeExp*
T0*#
_output_shapes
:?????????
g
gradients/pi/Sum_3_grad/ShapeConst*
valueB:*
_output_shapes
:*
dtype0
?
gradients/pi/Sum_3_grad/SizeConst*
_output_shapes
: *
value	B :*
dtype0*0
_class&
$"loc:@gradients/pi/Sum_3_grad/Shape
?
gradients/pi/Sum_3_grad/addAddpi/Sum_3/reduction_indicesgradients/pi/Sum_3_grad/Size*
_output_shapes
: *0
_class&
$"loc:@gradients/pi/Sum_3_grad/Shape*
T0
?
gradients/pi/Sum_3_grad/modFloorModgradients/pi/Sum_3_grad/addgradients/pi/Sum_3_grad/Size*
_output_shapes
: *0
_class&
$"loc:@gradients/pi/Sum_3_grad/Shape*
T0
?
gradients/pi/Sum_3_grad/Shape_1Const*0
_class&
$"loc:@gradients/pi/Sum_3_grad/Shape*
_output_shapes
: *
dtype0*
valueB 
?
#gradients/pi/Sum_3_grad/range/startConst*
dtype0*
_output_shapes
: *0
_class&
$"loc:@gradients/pi/Sum_3_grad/Shape*
value	B : 
?
#gradients/pi/Sum_3_grad/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :*0
_class&
$"loc:@gradients/pi/Sum_3_grad/Shape
?
gradients/pi/Sum_3_grad/rangeRange#gradients/pi/Sum_3_grad/range/startgradients/pi/Sum_3_grad/Size#gradients/pi/Sum_3_grad/range/delta*0
_class&
$"loc:@gradients/pi/Sum_3_grad/Shape*

Tidx0*
_output_shapes
:
?
"gradients/pi/Sum_3_grad/Fill/valueConst*0
_class&
$"loc:@gradients/pi/Sum_3_grad/Shape*
_output_shapes
: *
value	B :*
dtype0
?
gradients/pi/Sum_3_grad/FillFillgradients/pi/Sum_3_grad/Shape_1"gradients/pi/Sum_3_grad/Fill/value*0
_class&
$"loc:@gradients/pi/Sum_3_grad/Shape*

index_type0*
T0*
_output_shapes
: 
?
%gradients/pi/Sum_3_grad/DynamicStitchDynamicStitchgradients/pi/Sum_3_grad/rangegradients/pi/Sum_3_grad/modgradients/pi/Sum_3_grad/Shapegradients/pi/Sum_3_grad/Fill*0
_class&
$"loc:@gradients/pi/Sum_3_grad/Shape*
_output_shapes
:*
T0*
N
?
!gradients/pi/Sum_3_grad/Maximum/yConst*
dtype0*0
_class&
$"loc:@gradients/pi/Sum_3_grad/Shape*
_output_shapes
: *
value	B :
?
gradients/pi/Sum_3_grad/MaximumMaximum%gradients/pi/Sum_3_grad/DynamicStitch!gradients/pi/Sum_3_grad/Maximum/y*
_output_shapes
:*0
_class&
$"loc:@gradients/pi/Sum_3_grad/Shape*
T0
?
 gradients/pi/Sum_3_grad/floordivFloorDivgradients/pi/Sum_3_grad/Shapegradients/pi/Sum_3_grad/Maximum*
T0*
_output_shapes
:*0
_class&
$"loc:@gradients/pi/Sum_3_grad/Shape
?
gradients/pi/Sum_3_grad/ReshapeReshape gradients/pi/Mean_1_grad/truediv%gradients/pi/Sum_3_grad/DynamicStitch*
T0*
Tshape0*
_output_shapes
:
?
gradients/pi/Sum_3_grad/TileTilegradients/pi/Sum_3_grad/Reshape gradients/pi/Sum_3_grad/floordiv*
T0*
_output_shapes
:*

Tmultiples0
^
gradients/sub_grad/ShapeShapepi/Sum*
out_type0*
T0*
_output_shapes
:
g
gradients/sub_grad/Shape_1ShapePlaceholder_6*
_output_shapes
:*
T0*
out_type0
?
(gradients/sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/sub_grad/Shapegradients/sub_grad/Shape_1*2
_output_shapes 
:?????????:?????????*
T0
?
gradients/sub_grad/SumSumgradients/Exp_grad/mul(gradients/sub_grad/BroadcastGradientArgs*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
?
gradients/sub_grad/ReshapeReshapegradients/sub_grad/Sumgradients/sub_grad/Shape*
T0*#
_output_shapes
:?????????*
Tshape0
?
gradients/sub_grad/Sum_1Sumgradients/Exp_grad/mul*gradients/sub_grad/BroadcastGradientArgs:1*
	keep_dims( *
_output_shapes
:*

Tidx0*
T0
Z
gradients/sub_grad/NegNeggradients/sub_grad/Sum_1*
_output_shapes
:*
T0
?
gradients/sub_grad/Reshape_1Reshapegradients/sub_grad/Neggradients/sub_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:?????????
h
gradients/pi/add_10_grad/ShapeConst*
valueB:*
dtype0*
_output_shapes
:
c
 gradients/pi/add_10_grad/Shape_1Const*
valueB *
_output_shapes
: *
dtype0
?
.gradients/pi/add_10_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/pi/add_10_grad/Shape gradients/pi/add_10_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
gradients/pi/add_10_grad/SumSumgradients/pi/Sum_3_grad/Tile.gradients/pi/add_10_grad/BroadcastGradientArgs*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
?
 gradients/pi/add_10_grad/ReshapeReshapegradients/pi/add_10_grad/Sumgradients/pi/add_10_grad/Shape*
_output_shapes
:*
Tshape0*
T0
?
gradients/pi/add_10_grad/Sum_1Sumgradients/pi/Sum_3_grad/Tile0gradients/pi/add_10_grad/BroadcastGradientArgs:1*

Tidx0*
T0*
	keep_dims( *
_output_shapes
: 
?
"gradients/pi/add_10_grad/Reshape_1Reshapegradients/pi/add_10_grad/Sum_1 gradients/pi/add_10_grad/Shape_1*
T0*
_output_shapes
: *
Tshape0
c
gradients/pi/Sum_grad/ShapeShapepi/mul_2*
_output_shapes
:*
out_type0*
T0
?
gradients/pi/Sum_grad/SizeConst*
dtype0*
_output_shapes
: *.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
value	B :
?
gradients/pi/Sum_grad/addAddpi/Sum/reduction_indicesgradients/pi/Sum_grad/Size*
_output_shapes
: *
T0*.
_class$
" loc:@gradients/pi/Sum_grad/Shape
?
gradients/pi/Sum_grad/modFloorModgradients/pi/Sum_grad/addgradients/pi/Sum_grad/Size*.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
T0*
_output_shapes
: 
?
gradients/pi/Sum_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB *.
_class$
" loc:@gradients/pi/Sum_grad/Shape
?
!gradients/pi/Sum_grad/range/startConst*
value	B : *.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
dtype0*
_output_shapes
: 
?
!gradients/pi/Sum_grad/range/deltaConst*
dtype0*
_output_shapes
: *.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
value	B :
?
gradients/pi/Sum_grad/rangeRange!gradients/pi/Sum_grad/range/startgradients/pi/Sum_grad/Size!gradients/pi/Sum_grad/range/delta*
_output_shapes
:*

Tidx0*.
_class$
" loc:@gradients/pi/Sum_grad/Shape
?
 gradients/pi/Sum_grad/Fill/valueConst*.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
?
gradients/pi/Sum_grad/FillFillgradients/pi/Sum_grad/Shape_1 gradients/pi/Sum_grad/Fill/value*
_output_shapes
: *.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
T0*

index_type0
?
#gradients/pi/Sum_grad/DynamicStitchDynamicStitchgradients/pi/Sum_grad/rangegradients/pi/Sum_grad/modgradients/pi/Sum_grad/Shapegradients/pi/Sum_grad/Fill*
T0*
N*
_output_shapes
:*.
_class$
" loc:@gradients/pi/Sum_grad/Shape
?
gradients/pi/Sum_grad/Maximum/yConst*
dtype0*.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
value	B :*
_output_shapes
: 
?
gradients/pi/Sum_grad/MaximumMaximum#gradients/pi/Sum_grad/DynamicStitchgradients/pi/Sum_grad/Maximum/y*.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
_output_shapes
:*
T0
?
gradients/pi/Sum_grad/floordivFloorDivgradients/pi/Sum_grad/Shapegradients/pi/Sum_grad/Maximum*.
_class$
" loc:@gradients/pi/Sum_grad/Shape*
_output_shapes
:*
T0
?
gradients/pi/Sum_grad/ReshapeReshapegradients/sub_grad/Reshape#gradients/pi/Sum_grad/DynamicStitch*0
_output_shapes
:??????????????????*
T0*
Tshape0
?
gradients/pi/Sum_grad/TileTilegradients/pi/Sum_grad/Reshapegradients/pi/Sum_grad/floordiv*
T0*

Tmultiples0*'
_output_shapes
:?????????
`
gradients/pi/mul_2_grad/ShapeConst*
valueB *
_output_shapes
: *
dtype0
g
gradients/pi/mul_2_grad/Shape_1Shapepi/add_3*
out_type0*
T0*
_output_shapes
:
?
-gradients/pi/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/pi/mul_2_grad/Shapegradients/pi/mul_2_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
z
gradients/pi/mul_2_grad/MulMulgradients/pi/Sum_grad/Tilepi/add_3*
T0*'
_output_shapes
:?????????
?
gradients/pi/mul_2_grad/SumSumgradients/pi/mul_2_grad/Mul-gradients/pi/mul_2_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
gradients/pi/mul_2_grad/ReshapeReshapegradients/pi/mul_2_grad/Sumgradients/pi/mul_2_grad/Shape*
Tshape0*
_output_shapes
: *
T0
~
gradients/pi/mul_2_grad/Mul_1Mul
pi/mul_2/xgradients/pi/Sum_grad/Tile*'
_output_shapes
:?????????*
T0
?
gradients/pi/mul_2_grad/Sum_1Sumgradients/pi/mul_2_grad/Mul_1/gradients/pi/mul_2_grad/BroadcastGradientArgs:1*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
?
!gradients/pi/mul_2_grad/Reshape_1Reshapegradients/pi/mul_2_grad/Sum_1gradients/pi/mul_2_grad/Shape_1*'
_output_shapes
:?????????*
T0*
Tshape0
e
gradients/pi/add_3_grad/ShapeShapepi/add_2*
T0*
_output_shapes
:*
out_type0
b
gradients/pi/add_3_grad/Shape_1Const*
valueB *
_output_shapes
: *
dtype0
?
-gradients/pi/add_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/pi/add_3_grad/Shapegradients/pi/add_3_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
gradients/pi/add_3_grad/SumSum!gradients/pi/mul_2_grad/Reshape_1-gradients/pi/add_3_grad/BroadcastGradientArgs*
_output_shapes
:*
T0*
	keep_dims( *

Tidx0
?
gradients/pi/add_3_grad/ReshapeReshapegradients/pi/add_3_grad/Sumgradients/pi/add_3_grad/Shape*'
_output_shapes
:?????????*
T0*
Tshape0
?
gradients/pi/add_3_grad/Sum_1Sum!gradients/pi/mul_2_grad/Reshape_1/gradients/pi/add_3_grad/BroadcastGradientArgs:1*

Tidx0*
T0*
_output_shapes
:*
	keep_dims( 
?
!gradients/pi/add_3_grad/Reshape_1Reshapegradients/pi/add_3_grad/Sum_1gradients/pi/add_3_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
c
gradients/pi/add_2_grad/ShapeShapepi/pow*
T0*
_output_shapes
:*
out_type0
i
gradients/pi/add_2_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:
?
-gradients/pi/add_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/pi/add_2_grad/Shapegradients/pi/add_2_grad/Shape_1*2
_output_shapes 
:?????????:?????????*
T0
?
gradients/pi/add_2_grad/SumSumgradients/pi/add_3_grad/Reshape-gradients/pi/add_2_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
?
gradients/pi/add_2_grad/ReshapeReshapegradients/pi/add_2_grad/Sumgradients/pi/add_2_grad/Shape*
Tshape0*
T0*'
_output_shapes
:?????????
?
gradients/pi/add_2_grad/Sum_1Sumgradients/pi/add_3_grad/Reshape/gradients/pi/add_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
T0*
	keep_dims( 
?
!gradients/pi/add_2_grad/Reshape_1Reshapegradients/pi/add_2_grad/Sum_1gradients/pi/add_2_grad/Shape_1*
Tshape0*
_output_shapes
:*
T0
e
gradients/pi/pow_grad/ShapeShape
pi/truediv*
_output_shapes
:*
out_type0*
T0
`
gradients/pi/pow_grad/Shape_1Const*
dtype0*
valueB *
_output_shapes
: 
?
+gradients/pi/pow_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/pi/pow_grad/Shapegradients/pi/pow_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
}
gradients/pi/pow_grad/mulMulgradients/pi/add_2_grad/Reshapepi/pow/y*'
_output_shapes
:?????????*
T0
`
gradients/pi/pow_grad/sub/yConst*
dtype0*
valueB
 *  ??*
_output_shapes
: 
h
gradients/pi/pow_grad/subSubpi/pow/ygradients/pi/pow_grad/sub/y*
T0*
_output_shapes
: 
y
gradients/pi/pow_grad/PowPow
pi/truedivgradients/pi/pow_grad/sub*
T0*'
_output_shapes
:?????????
?
gradients/pi/pow_grad/mul_1Mulgradients/pi/pow_grad/mulgradients/pi/pow_grad/Pow*
T0*'
_output_shapes
:?????????
?
gradients/pi/pow_grad/SumSumgradients/pi/pow_grad/mul_1+gradients/pi/pow_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
gradients/pi/pow_grad/ReshapeReshapegradients/pi/pow_grad/Sumgradients/pi/pow_grad/Shape*'
_output_shapes
:?????????*
Tshape0*
T0
d
gradients/pi/pow_grad/Greater/yConst*
dtype0*
_output_shapes
: *
valueB
 *    
?
gradients/pi/pow_grad/GreaterGreater
pi/truedivgradients/pi/pow_grad/Greater/y*
T0*'
_output_shapes
:?????????
o
%gradients/pi/pow_grad/ones_like/ShapeShape
pi/truediv*
out_type0*
_output_shapes
:*
T0
j
%gradients/pi/pow_grad/ones_like/ConstConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
gradients/pi/pow_grad/ones_likeFill%gradients/pi/pow_grad/ones_like/Shape%gradients/pi/pow_grad/ones_like/Const*
T0*

index_type0*'
_output_shapes
:?????????
?
gradients/pi/pow_grad/SelectSelectgradients/pi/pow_grad/Greater
pi/truedivgradients/pi/pow_grad/ones_like*'
_output_shapes
:?????????*
T0
p
gradients/pi/pow_grad/LogLoggradients/pi/pow_grad/Select*
T0*'
_output_shapes
:?????????
k
 gradients/pi/pow_grad/zeros_like	ZerosLike
pi/truediv*
T0*'
_output_shapes
:?????????
?
gradients/pi/pow_grad/Select_1Selectgradients/pi/pow_grad/Greatergradients/pi/pow_grad/Log gradients/pi/pow_grad/zeros_like*'
_output_shapes
:?????????*
T0
}
gradients/pi/pow_grad/mul_2Mulgradients/pi/add_2_grad/Reshapepi/pow*'
_output_shapes
:?????????*
T0
?
gradients/pi/pow_grad/mul_3Mulgradients/pi/pow_grad/mul_2gradients/pi/pow_grad/Select_1*'
_output_shapes
:?????????*
T0
?
gradients/pi/pow_grad/Sum_1Sumgradients/pi/pow_grad/mul_3-gradients/pi/pow_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
_output_shapes
:*
T0
?
gradients/pi/pow_grad/Reshape_1Reshapegradients/pi/pow_grad/Sum_1gradients/pi/pow_grad/Shape_1*
T0*
_output_shapes
: *
Tshape0
`
gradients/pi/mul_1_grad/ShapeConst*
valueB *
_output_shapes
: *
dtype0
i
gradients/pi/mul_1_grad/Shape_1Const*
valueB:*
_output_shapes
:*
dtype0
?
-gradients/pi/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/pi/mul_1_grad/Shapegradients/pi/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
{
gradients/pi/mul_1_grad/MulMul!gradients/pi/add_2_grad/Reshape_1pi/log_std/read*
_output_shapes
:*
T0
?
gradients/pi/mul_1_grad/SumSumgradients/pi/mul_1_grad/Mul-gradients/pi/mul_1_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
?
gradients/pi/mul_1_grad/ReshapeReshapegradients/pi/mul_1_grad/Sumgradients/pi/mul_1_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
x
gradients/pi/mul_1_grad/Mul_1Mul
pi/mul_1/x!gradients/pi/add_2_grad/Reshape_1*
T0*
_output_shapes
:
?
gradients/pi/mul_1_grad/Sum_1Sumgradients/pi/mul_1_grad/Mul_1/gradients/pi/mul_1_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
_output_shapes
:*
T0
?
!gradients/pi/mul_1_grad/Reshape_1Reshapegradients/pi/mul_1_grad/Sum_1gradients/pi/mul_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes
:
e
gradients/pi/truediv_grad/ShapeShapepi/sub*
T0*
_output_shapes
:*
out_type0
k
!gradients/pi/truediv_grad/Shape_1Const*
valueB:*
_output_shapes
:*
dtype0
?
/gradients/pi/truediv_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/pi/truediv_grad/Shape!gradients/pi/truediv_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
!gradients/pi/truediv_grad/RealDivRealDivgradients/pi/pow_grad/Reshapepi/add_1*
T0*'
_output_shapes
:?????????
?
gradients/pi/truediv_grad/SumSum!gradients/pi/truediv_grad/RealDiv/gradients/pi/truediv_grad/BroadcastGradientArgs*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
?
!gradients/pi/truediv_grad/ReshapeReshapegradients/pi/truediv_grad/Sumgradients/pi/truediv_grad/Shape*
Tshape0*'
_output_shapes
:?????????*
T0
^
gradients/pi/truediv_grad/NegNegpi/sub*'
_output_shapes
:?????????*
T0
?
#gradients/pi/truediv_grad/RealDiv_1RealDivgradients/pi/truediv_grad/Negpi/add_1*'
_output_shapes
:?????????*
T0
?
#gradients/pi/truediv_grad/RealDiv_2RealDiv#gradients/pi/truediv_grad/RealDiv_1pi/add_1*'
_output_shapes
:?????????*
T0
?
gradients/pi/truediv_grad/mulMulgradients/pi/pow_grad/Reshape#gradients/pi/truediv_grad/RealDiv_2*
T0*'
_output_shapes
:?????????
?
gradients/pi/truediv_grad/Sum_1Sumgradients/pi/truediv_grad/mul1gradients/pi/truediv_grad/BroadcastGradientArgs:1*
	keep_dims( *
_output_shapes
:*

Tidx0*
T0
?
#gradients/pi/truediv_grad/Reshape_1Reshapegradients/pi/truediv_grad/Sum_1!gradients/pi/truediv_grad/Shape_1*
Tshape0*
T0*
_output_shapes
:
h
gradients/pi/sub_grad/ShapeShapePlaceholder_1*
out_type0*
T0*
_output_shapes
:
o
gradients/pi/sub_grad/Shape_1Shapepi/dense_2/BiasAdd*
out_type0*
_output_shapes
:*
T0
?
+gradients/pi/sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/pi/sub_grad/Shapegradients/pi/sub_grad/Shape_1*2
_output_shapes 
:?????????:?????????*
T0
?
gradients/pi/sub_grad/SumSum!gradients/pi/truediv_grad/Reshape+gradients/pi/sub_grad/BroadcastGradientArgs*

Tidx0*
	keep_dims( *
T0*
_output_shapes
:
?
gradients/pi/sub_grad/ReshapeReshapegradients/pi/sub_grad/Sumgradients/pi/sub_grad/Shape*'
_output_shapes
:?????????*
Tshape0*
T0
?
gradients/pi/sub_grad/Sum_1Sum!gradients/pi/truediv_grad/Reshape-gradients/pi/sub_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
`
gradients/pi/sub_grad/NegNeggradients/pi/sub_grad/Sum_1*
T0*
_output_shapes
:
?
gradients/pi/sub_grad/Reshape_1Reshapegradients/pi/sub_grad/Neggradients/pi/sub_grad/Shape_1*'
_output_shapes
:?????????*
T0*
Tshape0
g
gradients/pi/add_1_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB:
b
gradients/pi/add_1_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
?
-gradients/pi/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/pi/add_1_grad/Shapegradients/pi/add_1_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
gradients/pi/add_1_grad/SumSum#gradients/pi/truediv_grad/Reshape_1-gradients/pi/add_1_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0
?
gradients/pi/add_1_grad/ReshapeReshapegradients/pi/add_1_grad/Sumgradients/pi/add_1_grad/Shape*
T0*
Tshape0*
_output_shapes
:
?
gradients/pi/add_1_grad/Sum_1Sum#gradients/pi/truediv_grad/Reshape_1/gradients/pi/add_1_grad/BroadcastGradientArgs:1*
T0*
	keep_dims( *
_output_shapes
: *

Tidx0
?
!gradients/pi/add_1_grad/Reshape_1Reshapegradients/pi/add_1_grad/Sum_1gradients/pi/add_1_grad/Shape_1*
Tshape0*
_output_shapes
: *
T0
?
-gradients/pi/dense_2/BiasAdd_grad/BiasAddGradBiasAddGradgradients/pi/sub_grad/Reshape_1*
_output_shapes
:*
T0*
data_formatNHWC
r
gradients/pi/Exp_1_grad/mulMulgradients/pi/add_1_grad/Reshapepi/Exp_1*
_output_shapes
:*
T0
?
'gradients/pi/dense_2/MatMul_grad/MatMulMatMulgradients/pi/sub_grad/Reshape_1pi/dense_2/kernel/read*
T0*(
_output_shapes
:??????????*
transpose_b(*
transpose_a( 
?
)gradients/pi/dense_2/MatMul_grad/MatMul_1MatMulpi/dense_1/Tanhgradients/pi/sub_grad/Reshape_1*
transpose_a(*
T0*
_output_shapes
:	?*
transpose_b( 
?
gradients/AddNAddN gradients/pi/add_10_grad/Reshape!gradients/pi/mul_1_grad/Reshape_1gradients/pi/Exp_1_grad/mul*
T0*3
_class)
'%loc:@gradients/pi/add_10_grad/Reshape*
N*
_output_shapes
:
?
'gradients/pi/dense_1/Tanh_grad/TanhGradTanhGradpi/dense_1/Tanh'gradients/pi/dense_2/MatMul_grad/MatMul*(
_output_shapes
:??????????*
T0
?
-gradients/pi/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad'gradients/pi/dense_1/Tanh_grad/TanhGrad*
_output_shapes	
:?*
T0*
data_formatNHWC
?
'gradients/pi/dense_1/MatMul_grad/MatMulMatMul'gradients/pi/dense_1/Tanh_grad/TanhGradpi/dense_1/kernel/read*(
_output_shapes
:??????????*
transpose_b(*
transpose_a( *
T0
?
)gradients/pi/dense_1/MatMul_grad/MatMul_1MatMulpi/dense/Tanh'gradients/pi/dense_1/Tanh_grad/TanhGrad*
transpose_b( *
T0*
transpose_a(* 
_output_shapes
:
??
?
%gradients/pi/dense/Tanh_grad/TanhGradTanhGradpi/dense/Tanh'gradients/pi/dense_1/MatMul_grad/MatMul*(
_output_shapes
:??????????*
T0
?
+gradients/pi/dense/BiasAdd_grad/BiasAddGradBiasAddGrad%gradients/pi/dense/Tanh_grad/TanhGrad*
data_formatNHWC*
_output_shapes	
:?*
T0
?
%gradients/pi/dense/MatMul_grad/MatMulMatMul%gradients/pi/dense/Tanh_grad/TanhGradpi/dense/kernel/read*'
_output_shapes
:?????????<*
transpose_a( *
transpose_b(*
T0
?
'gradients/pi/dense/MatMul_grad/MatMul_1MatMulPlaceholder%gradients/pi/dense/Tanh_grad/TanhGrad*
_output_shapes
:	<?*
transpose_a(*
T0*
transpose_b( 
`
Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:
?????????
~
ReshapeReshape'gradients/pi/dense/MatMul_grad/MatMul_1Reshape/shape*
T0*
_output_shapes	
:?x*
Tshape0
b
Reshape_1/shapeConst*
valueB:
?????????*
dtype0*
_output_shapes
:
?
	Reshape_1Reshape+gradients/pi/dense/BiasAdd_grad/BiasAddGradReshape_1/shape*
_output_shapes	
:?*
Tshape0*
T0
b
Reshape_2/shapeConst*
dtype0*
valueB:
?????????*
_output_shapes
:
?
	Reshape_2Reshape)gradients/pi/dense_1/MatMul_grad/MatMul_1Reshape_2/shape*
_output_shapes

:??*
T0*
Tshape0
b
Reshape_3/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
?
	Reshape_3Reshape-gradients/pi/dense_1/BiasAdd_grad/BiasAddGradReshape_3/shape*
T0*
Tshape0*
_output_shapes	
:?
b
Reshape_4/shapeConst*
dtype0*
valueB:
?????????*
_output_shapes
:
?
	Reshape_4Reshape)gradients/pi/dense_2/MatMul_grad/MatMul_1Reshape_4/shape*
T0*
_output_shapes	
:?*
Tshape0
b
Reshape_5/shapeConst*
_output_shapes
:*
valueB:
?????????*
dtype0
?
	Reshape_5Reshape-gradients/pi/dense_2/BiasAdd_grad/BiasAddGradReshape_5/shape*
Tshape0*
T0*
_output_shapes
:
b
Reshape_6/shapeConst*
_output_shapes
:*
valueB:
?????????*
dtype0
h
	Reshape_6Reshapegradients/AddNReshape_6/shape*
_output_shapes
:*
Tshape0*
T0
M
concat/axisConst*
_output_shapes
: *
value	B : *
dtype0
?
concatConcatV2Reshape	Reshape_1	Reshape_2	Reshape_3	Reshape_4	Reshape_5	Reshape_6concat/axis*

Tidx0*
N*
_output_shapes

:??*
T0
T
gradients_1/ShapeConst*
valueB *
_output_shapes
: *
dtype0
Z
gradients_1/grad_ys_0Const*
dtype0*
valueB
 *  ??*
_output_shapes
: 
u
gradients_1/FillFillgradients_1/Shapegradients_1/grad_ys_0*
_output_shapes
: *
T0*

index_type0
p
&gradients_1/pi/Mean_grad/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?
 gradients_1/pi/Mean_grad/ReshapeReshapegradients_1/Fill&gradients_1/pi/Mean_grad/Reshape/shape*
_output_shapes
:*
T0*
Tshape0
f
gradients_1/pi/Mean_grad/ShapeShapepi/Sum_2*
_output_shapes
:*
T0*
out_type0
?
gradients_1/pi/Mean_grad/TileTile gradients_1/pi/Mean_grad/Reshapegradients_1/pi/Mean_grad/Shape*
T0*#
_output_shapes
:?????????*

Tmultiples0
h
 gradients_1/pi/Mean_grad/Shape_1Shapepi/Sum_2*
_output_shapes
:*
out_type0*
T0
c
 gradients_1/pi/Mean_grad/Shape_2Const*
_output_shapes
: *
dtype0*
valueB 
h
gradients_1/pi/Mean_grad/ConstConst*
dtype0*
valueB: *
_output_shapes
:
?
gradients_1/pi/Mean_grad/ProdProd gradients_1/pi/Mean_grad/Shape_1gradients_1/pi/Mean_grad/Const*
_output_shapes
: *

Tidx0*
	keep_dims( *
T0
j
 gradients_1/pi/Mean_grad/Const_1Const*
dtype0*
_output_shapes
:*
valueB: 
?
gradients_1/pi/Mean_grad/Prod_1Prod gradients_1/pi/Mean_grad/Shape_2 gradients_1/pi/Mean_grad/Const_1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
: 
d
"gradients_1/pi/Mean_grad/Maximum/yConst*
_output_shapes
: *
value	B :*
dtype0
?
 gradients_1/pi/Mean_grad/MaximumMaximumgradients_1/pi/Mean_grad/Prod_1"gradients_1/pi/Mean_grad/Maximum/y*
_output_shapes
: *
T0
?
!gradients_1/pi/Mean_grad/floordivFloorDivgradients_1/pi/Mean_grad/Prod gradients_1/pi/Mean_grad/Maximum*
_output_shapes
: *
T0
?
gradients_1/pi/Mean_grad/CastCast!gradients_1/pi/Mean_grad/floordiv*
_output_shapes
: *

SrcT0*
Truncate( *

DstT0
?
 gradients_1/pi/Mean_grad/truedivRealDivgradients_1/pi/Mean_grad/Tilegradients_1/pi/Mean_grad/Cast*
T0*#
_output_shapes
:?????????
g
gradients_1/pi/Sum_2_grad/ShapeShapepi/sub_4*
T0*
out_type0*
_output_shapes
:
?
gradients_1/pi/Sum_2_grad/SizeConst*
value	B :*2
_class(
&$loc:@gradients_1/pi/Sum_2_grad/Shape*
_output_shapes
: *
dtype0
?
gradients_1/pi/Sum_2_grad/addAddpi/Sum_2/reduction_indicesgradients_1/pi/Sum_2_grad/Size*2
_class(
&$loc:@gradients_1/pi/Sum_2_grad/Shape*
T0*
_output_shapes
: 
?
gradients_1/pi/Sum_2_grad/modFloorModgradients_1/pi/Sum_2_grad/addgradients_1/pi/Sum_2_grad/Size*2
_class(
&$loc:@gradients_1/pi/Sum_2_grad/Shape*
_output_shapes
: *
T0
?
!gradients_1/pi/Sum_2_grad/Shape_1Const*
_output_shapes
: *
valueB *
dtype0*2
_class(
&$loc:@gradients_1/pi/Sum_2_grad/Shape
?
%gradients_1/pi/Sum_2_grad/range/startConst*2
_class(
&$loc:@gradients_1/pi/Sum_2_grad/Shape*
_output_shapes
: *
dtype0*
value	B : 
?
%gradients_1/pi/Sum_2_grad/range/deltaConst*
_output_shapes
: *
dtype0*2
_class(
&$loc:@gradients_1/pi/Sum_2_grad/Shape*
value	B :
?
gradients_1/pi/Sum_2_grad/rangeRange%gradients_1/pi/Sum_2_grad/range/startgradients_1/pi/Sum_2_grad/Size%gradients_1/pi/Sum_2_grad/range/delta*

Tidx0*2
_class(
&$loc:@gradients_1/pi/Sum_2_grad/Shape*
_output_shapes
:
?
$gradients_1/pi/Sum_2_grad/Fill/valueConst*
dtype0*
value	B :*2
_class(
&$loc:@gradients_1/pi/Sum_2_grad/Shape*
_output_shapes
: 
?
gradients_1/pi/Sum_2_grad/FillFill!gradients_1/pi/Sum_2_grad/Shape_1$gradients_1/pi/Sum_2_grad/Fill/value*
_output_shapes
: *
T0*2
_class(
&$loc:@gradients_1/pi/Sum_2_grad/Shape*

index_type0
?
'gradients_1/pi/Sum_2_grad/DynamicStitchDynamicStitchgradients_1/pi/Sum_2_grad/rangegradients_1/pi/Sum_2_grad/modgradients_1/pi/Sum_2_grad/Shapegradients_1/pi/Sum_2_grad/Fill*2
_class(
&$loc:@gradients_1/pi/Sum_2_grad/Shape*
_output_shapes
:*
T0*
N
?
#gradients_1/pi/Sum_2_grad/Maximum/yConst*
dtype0*
value	B :*
_output_shapes
: *2
_class(
&$loc:@gradients_1/pi/Sum_2_grad/Shape
?
!gradients_1/pi/Sum_2_grad/MaximumMaximum'gradients_1/pi/Sum_2_grad/DynamicStitch#gradients_1/pi/Sum_2_grad/Maximum/y*2
_class(
&$loc:@gradients_1/pi/Sum_2_grad/Shape*
_output_shapes
:*
T0
?
"gradients_1/pi/Sum_2_grad/floordivFloorDivgradients_1/pi/Sum_2_grad/Shape!gradients_1/pi/Sum_2_grad/Maximum*
_output_shapes
:*2
_class(
&$loc:@gradients_1/pi/Sum_2_grad/Shape*
T0
?
!gradients_1/pi/Sum_2_grad/ReshapeReshape gradients_1/pi/Mean_grad/truediv'gradients_1/pi/Sum_2_grad/DynamicStitch*
Tshape0*
T0*0
_output_shapes
:??????????????????
?
gradients_1/pi/Sum_2_grad/TileTile!gradients_1/pi/Sum_2_grad/Reshape"gradients_1/pi/Sum_2_grad/floordiv*
T0*'
_output_shapes
:?????????*

Tmultiples0
g
gradients_1/pi/sub_4_grad/ShapeShapepi/add_9*
T0*
_output_shapes
:*
out_type0
k
!gradients_1/pi/sub_4_grad/Shape_1Const*
_output_shapes
:*
valueB:*
dtype0
?
/gradients_1/pi/sub_4_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/pi/sub_4_grad/Shape!gradients_1/pi/sub_4_grad/Shape_1*2
_output_shapes 
:?????????:?????????*
T0
?
gradients_1/pi/sub_4_grad/SumSumgradients_1/pi/Sum_2_grad/Tile/gradients_1/pi/sub_4_grad/BroadcastGradientArgs*

Tidx0*
_output_shapes
:*
	keep_dims( *
T0
?
!gradients_1/pi/sub_4_grad/ReshapeReshapegradients_1/pi/sub_4_grad/Sumgradients_1/pi/sub_4_grad/Shape*
Tshape0*
T0*'
_output_shapes
:?????????
?
gradients_1/pi/sub_4_grad/Sum_1Sumgradients_1/pi/Sum_2_grad/Tile1gradients_1/pi/sub_4_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
h
gradients_1/pi/sub_4_grad/NegNeggradients_1/pi/sub_4_grad/Sum_1*
_output_shapes
:*
T0
?
#gradients_1/pi/sub_4_grad/Reshape_1Reshapegradients_1/pi/sub_4_grad/Neg!gradients_1/pi/sub_4_grad/Shape_1*
T0*
_output_shapes
:*
Tshape0
g
gradients_1/pi/add_9_grad/ShapeShapepi/mul_7*
T0*
_output_shapes
:*
out_type0
q
!gradients_1/pi/add_9_grad/Shape_1Shapepi/Placeholder_1*
_output_shapes
:*
out_type0*
T0
?
/gradients_1/pi/add_9_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/pi/add_9_grad/Shape!gradients_1/pi/add_9_grad/Shape_1*2
_output_shapes 
:?????????:?????????*
T0
?
gradients_1/pi/add_9_grad/SumSum!gradients_1/pi/sub_4_grad/Reshape/gradients_1/pi/add_9_grad/BroadcastGradientArgs*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
?
!gradients_1/pi/add_9_grad/ReshapeReshapegradients_1/pi/add_9_grad/Sumgradients_1/pi/add_9_grad/Shape*
T0*
Tshape0*'
_output_shapes
:?????????
?
gradients_1/pi/add_9_grad/Sum_1Sum!gradients_1/pi/sub_4_grad/Reshape1gradients_1/pi/add_9_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
?
#gradients_1/pi/add_9_grad/Reshape_1Reshapegradients_1/pi/add_9_grad/Sum_1!gradients_1/pi/add_9_grad/Shape_1*
Tshape0*'
_output_shapes
:?????????*
T0
b
gradients_1/pi/mul_7_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
i
!gradients_1/pi/mul_7_grad/Shape_1Shapepi/sub_3*
T0*
out_type0*
_output_shapes
:
?
/gradients_1/pi/mul_7_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/pi/mul_7_grad/Shape!gradients_1/pi/mul_7_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
gradients_1/pi/mul_7_grad/MulMul!gradients_1/pi/add_9_grad/Reshapepi/sub_3*
T0*'
_output_shapes
:?????????
?
gradients_1/pi/mul_7_grad/SumSumgradients_1/pi/mul_7_grad/Mul/gradients_1/pi/mul_7_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
?
!gradients_1/pi/mul_7_grad/ReshapeReshapegradients_1/pi/mul_7_grad/Sumgradients_1/pi/mul_7_grad/Shape*
Tshape0*
_output_shapes
: *
T0
?
gradients_1/pi/mul_7_grad/Mul_1Mul
pi/mul_7/x!gradients_1/pi/add_9_grad/Reshape*
T0*'
_output_shapes
:?????????
?
gradients_1/pi/mul_7_grad/Sum_1Sumgradients_1/pi/mul_7_grad/Mul_11gradients_1/pi/mul_7_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
?
#gradients_1/pi/mul_7_grad/Reshape_1Reshapegradients_1/pi/mul_7_grad/Sum_1!gradients_1/pi/mul_7_grad/Shape_1*'
_output_shapes
:?????????*
T0*
Tshape0
k
gradients_1/pi/sub_3_grad/ShapeShapepi/truediv_2*
T0*
_output_shapes
:*
out_type0
d
!gradients_1/pi/sub_3_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
?
/gradients_1/pi/sub_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/pi/sub_3_grad/Shape!gradients_1/pi/sub_3_grad/Shape_1*2
_output_shapes 
:?????????:?????????*
T0
?
gradients_1/pi/sub_3_grad/SumSum#gradients_1/pi/mul_7_grad/Reshape_1/gradients_1/pi/sub_3_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
?
!gradients_1/pi/sub_3_grad/ReshapeReshapegradients_1/pi/sub_3_grad/Sumgradients_1/pi/sub_3_grad/Shape*
T0*
Tshape0*'
_output_shapes
:?????????
?
gradients_1/pi/sub_3_grad/Sum_1Sum#gradients_1/pi/mul_7_grad/Reshape_11gradients_1/pi/sub_3_grad/BroadcastGradientArgs:1*
	keep_dims( *
_output_shapes
:*

Tidx0*
T0
h
gradients_1/pi/sub_3_grad/NegNeggradients_1/pi/sub_3_grad/Sum_1*
T0*
_output_shapes
:
?
#gradients_1/pi/sub_3_grad/Reshape_1Reshapegradients_1/pi/sub_3_grad/Neg!gradients_1/pi/sub_3_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
k
#gradients_1/pi/truediv_2_grad/ShapeShapepi/add_7*
T0*
_output_shapes
:*
out_type0
m
%gradients_1/pi/truediv_2_grad/Shape_1Shapepi/add_8*
T0*
_output_shapes
:*
out_type0
?
3gradients_1/pi/truediv_2_grad/BroadcastGradientArgsBroadcastGradientArgs#gradients_1/pi/truediv_2_grad/Shape%gradients_1/pi/truediv_2_grad/Shape_1*2
_output_shapes 
:?????????:?????????*
T0
?
%gradients_1/pi/truediv_2_grad/RealDivRealDiv!gradients_1/pi/sub_3_grad/Reshapepi/add_8*'
_output_shapes
:?????????*
T0
?
!gradients_1/pi/truediv_2_grad/SumSum%gradients_1/pi/truediv_2_grad/RealDiv3gradients_1/pi/truediv_2_grad/BroadcastGradientArgs*
	keep_dims( *
_output_shapes
:*

Tidx0*
T0
?
%gradients_1/pi/truediv_2_grad/ReshapeReshape!gradients_1/pi/truediv_2_grad/Sum#gradients_1/pi/truediv_2_grad/Shape*'
_output_shapes
:?????????*
T0*
Tshape0
d
!gradients_1/pi/truediv_2_grad/NegNegpi/add_7*
T0*'
_output_shapes
:?????????
?
'gradients_1/pi/truediv_2_grad/RealDiv_1RealDiv!gradients_1/pi/truediv_2_grad/Negpi/add_8*
T0*'
_output_shapes
:?????????
?
'gradients_1/pi/truediv_2_grad/RealDiv_2RealDiv'gradients_1/pi/truediv_2_grad/RealDiv_1pi/add_8*'
_output_shapes
:?????????*
T0
?
!gradients_1/pi/truediv_2_grad/mulMul!gradients_1/pi/sub_3_grad/Reshape'gradients_1/pi/truediv_2_grad/RealDiv_2*
T0*'
_output_shapes
:?????????
?
#gradients_1/pi/truediv_2_grad/Sum_1Sum!gradients_1/pi/truediv_2_grad/mul5gradients_1/pi/truediv_2_grad/BroadcastGradientArgs:1*

Tidx0*
T0*
_output_shapes
:*
	keep_dims( 
?
'gradients_1/pi/truediv_2_grad/Reshape_1Reshape#gradients_1/pi/truediv_2_grad/Sum_1%gradients_1/pi/truediv_2_grad/Shape_1*
Tshape0*'
_output_shapes
:?????????*
T0
g
gradients_1/pi/add_7_grad/ShapeShapepi/pow_2*
_output_shapes
:*
out_type0*
T0
k
!gradients_1/pi/add_7_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:
?
/gradients_1/pi/add_7_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/pi/add_7_grad/Shape!gradients_1/pi/add_7_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
gradients_1/pi/add_7_grad/SumSum%gradients_1/pi/truediv_2_grad/Reshape/gradients_1/pi/add_7_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
?
!gradients_1/pi/add_7_grad/ReshapeReshapegradients_1/pi/add_7_grad/Sumgradients_1/pi/add_7_grad/Shape*
T0*'
_output_shapes
:?????????*
Tshape0
?
gradients_1/pi/add_7_grad/Sum_1Sum%gradients_1/pi/truediv_2_grad/Reshape1gradients_1/pi/add_7_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
?
#gradients_1/pi/add_7_grad/Reshape_1Reshapegradients_1/pi/add_7_grad/Sum_1!gradients_1/pi/add_7_grad/Shape_1*
Tshape0*
T0*
_output_shapes
:
g
gradients_1/pi/pow_2_grad/ShapeShapepi/sub_2*
out_type0*
_output_shapes
:*
T0
d
!gradients_1/pi/pow_2_grad/Shape_1Const*
valueB *
_output_shapes
: *
dtype0
?
/gradients_1/pi/pow_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/pi/pow_2_grad/Shape!gradients_1/pi/pow_2_grad/Shape_1*2
_output_shapes 
:?????????:?????????*
T0
?
gradients_1/pi/pow_2_grad/mulMul!gradients_1/pi/add_7_grad/Reshape
pi/pow_2/y*
T0*'
_output_shapes
:?????????
d
gradients_1/pi/pow_2_grad/sub/yConst*
valueB
 *  ??*
_output_shapes
: *
dtype0
r
gradients_1/pi/pow_2_grad/subSub
pi/pow_2/ygradients_1/pi/pow_2_grad/sub/y*
T0*
_output_shapes
: 

gradients_1/pi/pow_2_grad/PowPowpi/sub_2gradients_1/pi/pow_2_grad/sub*
T0*'
_output_shapes
:?????????
?
gradients_1/pi/pow_2_grad/mul_1Mulgradients_1/pi/pow_2_grad/mulgradients_1/pi/pow_2_grad/Pow*
T0*'
_output_shapes
:?????????
?
gradients_1/pi/pow_2_grad/SumSumgradients_1/pi/pow_2_grad/mul_1/gradients_1/pi/pow_2_grad/BroadcastGradientArgs*

Tidx0*
_output_shapes
:*
	keep_dims( *
T0
?
!gradients_1/pi/pow_2_grad/ReshapeReshapegradients_1/pi/pow_2_grad/Sumgradients_1/pi/pow_2_grad/Shape*'
_output_shapes
:?????????*
T0*
Tshape0
h
#gradients_1/pi/pow_2_grad/Greater/yConst*
_output_shapes
: *
dtype0*
valueB
 *    
?
!gradients_1/pi/pow_2_grad/GreaterGreaterpi/sub_2#gradients_1/pi/pow_2_grad/Greater/y*
T0*'
_output_shapes
:?????????
q
)gradients_1/pi/pow_2_grad/ones_like/ShapeShapepi/sub_2*
out_type0*
T0*
_output_shapes
:
n
)gradients_1/pi/pow_2_grad/ones_like/ConstConst*
dtype0*
valueB
 *  ??*
_output_shapes
: 
?
#gradients_1/pi/pow_2_grad/ones_likeFill)gradients_1/pi/pow_2_grad/ones_like/Shape)gradients_1/pi/pow_2_grad/ones_like/Const*

index_type0*'
_output_shapes
:?????????*
T0
?
 gradients_1/pi/pow_2_grad/SelectSelect!gradients_1/pi/pow_2_grad/Greaterpi/sub_2#gradients_1/pi/pow_2_grad/ones_like*'
_output_shapes
:?????????*
T0
x
gradients_1/pi/pow_2_grad/LogLog gradients_1/pi/pow_2_grad/Select*
T0*'
_output_shapes
:?????????
m
$gradients_1/pi/pow_2_grad/zeros_like	ZerosLikepi/sub_2*'
_output_shapes
:?????????*
T0
?
"gradients_1/pi/pow_2_grad/Select_1Select!gradients_1/pi/pow_2_grad/Greatergradients_1/pi/pow_2_grad/Log$gradients_1/pi/pow_2_grad/zeros_like*'
_output_shapes
:?????????*
T0
?
gradients_1/pi/pow_2_grad/mul_2Mul!gradients_1/pi/add_7_grad/Reshapepi/pow_2*'
_output_shapes
:?????????*
T0
?
gradients_1/pi/pow_2_grad/mul_3Mulgradients_1/pi/pow_2_grad/mul_2"gradients_1/pi/pow_2_grad/Select_1*'
_output_shapes
:?????????*
T0
?
gradients_1/pi/pow_2_grad/Sum_1Sumgradients_1/pi/pow_2_grad/mul_31gradients_1/pi/pow_2_grad/BroadcastGradientArgs:1*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
?
#gradients_1/pi/pow_2_grad/Reshape_1Reshapegradients_1/pi/pow_2_grad/Sum_1!gradients_1/pi/pow_2_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
x
gradients_1/pi/Exp_3_grad/mulMul#gradients_1/pi/add_7_grad/Reshape_1pi/Exp_3*
T0*
_output_shapes
:
m
gradients_1/pi/sub_2_grad/ShapeShapepi/Placeholder*
T0*
_output_shapes
:*
out_type0
s
!gradients_1/pi/sub_2_grad/Shape_1Shapepi/dense_2/BiasAdd*
_output_shapes
:*
out_type0*
T0
?
/gradients_1/pi/sub_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/pi/sub_2_grad/Shape!gradients_1/pi/sub_2_grad/Shape_1*2
_output_shapes 
:?????????:?????????*
T0
?
gradients_1/pi/sub_2_grad/SumSum!gradients_1/pi/pow_2_grad/Reshape/gradients_1/pi/sub_2_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *
T0*

Tidx0
?
!gradients_1/pi/sub_2_grad/ReshapeReshapegradients_1/pi/sub_2_grad/Sumgradients_1/pi/sub_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:?????????
?
gradients_1/pi/sub_2_grad/Sum_1Sum!gradients_1/pi/pow_2_grad/Reshape1gradients_1/pi/sub_2_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
h
gradients_1/pi/sub_2_grad/NegNeggradients_1/pi/sub_2_grad/Sum_1*
_output_shapes
:*
T0
?
#gradients_1/pi/sub_2_grad/Reshape_1Reshapegradients_1/pi/sub_2_grad/Neg!gradients_1/pi/sub_2_grad/Shape_1*
T0*'
_output_shapes
:?????????*
Tshape0
b
gradients_1/pi/mul_5_grad/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
k
!gradients_1/pi/mul_5_grad/Shape_1Const*
valueB:*
dtype0*
_output_shapes
:
?
/gradients_1/pi/mul_5_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/pi/mul_5_grad/Shape!gradients_1/pi/mul_5_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
y
gradients_1/pi/mul_5_grad/MulMulgradients_1/pi/Exp_3_grad/mulpi/log_std/read*
_output_shapes
:*
T0
?
gradients_1/pi/mul_5_grad/SumSumgradients_1/pi/mul_5_grad/Mul/gradients_1/pi/mul_5_grad/BroadcastGradientArgs*
	keep_dims( *
T0*
_output_shapes
: *

Tidx0
?
!gradients_1/pi/mul_5_grad/ReshapeReshapegradients_1/pi/mul_5_grad/Sumgradients_1/pi/mul_5_grad/Shape*
Tshape0*
_output_shapes
: *
T0
v
gradients_1/pi/mul_5_grad/Mul_1Mul
pi/mul_5/xgradients_1/pi/Exp_3_grad/mul*
_output_shapes
:*
T0
?
gradients_1/pi/mul_5_grad/Sum_1Sumgradients_1/pi/mul_5_grad/Mul_11gradients_1/pi/mul_5_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*

Tidx0*
	keep_dims( 
?
#gradients_1/pi/mul_5_grad/Reshape_1Reshapegradients_1/pi/mul_5_grad/Sum_1!gradients_1/pi/mul_5_grad/Shape_1*
_output_shapes
:*
Tshape0*
T0
?
/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad#gradients_1/pi/sub_2_grad/Reshape_1*
T0*
_output_shapes
:*
data_formatNHWC
?
gradients_1/AddNAddN#gradients_1/pi/sub_4_grad/Reshape_1#gradients_1/pi/mul_5_grad/Reshape_1*
N*
_output_shapes
:*
T0*6
_class,
*(loc:@gradients_1/pi/sub_4_grad/Reshape_1
?
)gradients_1/pi/dense_2/MatMul_grad/MatMulMatMul#gradients_1/pi/sub_2_grad/Reshape_1pi/dense_2/kernel/read*
T0*
transpose_a( *
transpose_b(*(
_output_shapes
:??????????
?
+gradients_1/pi/dense_2/MatMul_grad/MatMul_1MatMulpi/dense_1/Tanh#gradients_1/pi/sub_2_grad/Reshape_1*
T0*
transpose_b( *
transpose_a(*
_output_shapes
:	?
?
)gradients_1/pi/dense_1/Tanh_grad/TanhGradTanhGradpi/dense_1/Tanh)gradients_1/pi/dense_2/MatMul_grad/MatMul*(
_output_shapes
:??????????*
T0
?
/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad)gradients_1/pi/dense_1/Tanh_grad/TanhGrad*
T0*
data_formatNHWC*
_output_shapes	
:?
?
)gradients_1/pi/dense_1/MatMul_grad/MatMulMatMul)gradients_1/pi/dense_1/Tanh_grad/TanhGradpi/dense_1/kernel/read*
T0*
transpose_a( *(
_output_shapes
:??????????*
transpose_b(
?
+gradients_1/pi/dense_1/MatMul_grad/MatMul_1MatMulpi/dense/Tanh)gradients_1/pi/dense_1/Tanh_grad/TanhGrad*
transpose_b( *
transpose_a(* 
_output_shapes
:
??*
T0
?
'gradients_1/pi/dense/Tanh_grad/TanhGradTanhGradpi/dense/Tanh)gradients_1/pi/dense_1/MatMul_grad/MatMul*
T0*(
_output_shapes
:??????????
?
-gradients_1/pi/dense/BiasAdd_grad/BiasAddGradBiasAddGrad'gradients_1/pi/dense/Tanh_grad/TanhGrad*
T0*
data_formatNHWC*
_output_shapes	
:?
?
'gradients_1/pi/dense/MatMul_grad/MatMulMatMul'gradients_1/pi/dense/Tanh_grad/TanhGradpi/dense/kernel/read*
T0*
transpose_b(*
transpose_a( *'
_output_shapes
:?????????<
?
)gradients_1/pi/dense/MatMul_grad/MatMul_1MatMulPlaceholder'gradients_1/pi/dense/Tanh_grad/TanhGrad*
transpose_b( *
transpose_a(*
T0*
_output_shapes
:	<?
b
Reshape_7/shapeConst*
dtype0*
_output_shapes
:*
valueB:
?????????
?
	Reshape_7Reshape)gradients_1/pi/dense/MatMul_grad/MatMul_1Reshape_7/shape*
Tshape0*
T0*
_output_shapes	
:?x
b
Reshape_8/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
?
	Reshape_8Reshape-gradients_1/pi/dense/BiasAdd_grad/BiasAddGradReshape_8/shape*
Tshape0*
_output_shapes	
:?*
T0
b
Reshape_9/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
?
	Reshape_9Reshape+gradients_1/pi/dense_1/MatMul_grad/MatMul_1Reshape_9/shape*
Tshape0*
_output_shapes

:??*
T0
c
Reshape_10/shapeConst*
valueB:
?????????*
dtype0*
_output_shapes
:
?

Reshape_10Reshape/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGradReshape_10/shape*
T0*
Tshape0*
_output_shapes	
:?
c
Reshape_11/shapeConst*
valueB:
?????????*
dtype0*
_output_shapes
:
?

Reshape_11Reshape+gradients_1/pi/dense_2/MatMul_grad/MatMul_1Reshape_11/shape*
_output_shapes	
:?*
Tshape0*
T0
c
Reshape_12/shapeConst*
valueB:
?????????*
_output_shapes
:*
dtype0
?

Reshape_12Reshape/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGradReshape_12/shape*
T0*
Tshape0*
_output_shapes
:
c
Reshape_13/shapeConst*
valueB:
?????????*
_output_shapes
:*
dtype0
l

Reshape_13Reshapegradients_1/AddNReshape_13/shape*
Tshape0*
T0*
_output_shapes
:
O
concat_1/axisConst*
value	B : *
_output_shapes
: *
dtype0
?
concat_1ConcatV2	Reshape_7	Reshape_8	Reshape_9
Reshape_10
Reshape_11
Reshape_12
Reshape_13concat_1/axis*

Tidx0*
T0*
_output_shapes

:??*
N
Z
Placeholder_9Placeholder*
shape:??*
_output_shapes

:??*
dtype0
L
mul_3Mulconcat_1Placeholder_9*
_output_shapes

:??*
T0
Q
Const_3Const*
_output_shapes
:*
dtype0*
valueB: 
X
SumSummul_3Const_3*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
T
gradients_2/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
Z
gradients_2/grad_ys_0Const*
valueB
 *  ??*
_output_shapes
: *
dtype0
u
gradients_2/FillFillgradients_2/Shapegradients_2/grad_ys_0*
_output_shapes
: *
T0*

index_type0
l
"gradients_2/Sum_grad/Reshape/shapeConst*
dtype0*
valueB:*
_output_shapes
:
?
gradients_2/Sum_grad/ReshapeReshapegradients_2/Fill"gradients_2/Sum_grad/Reshape/shape*
Tshape0*
_output_shapes
:*
T0
f
gradients_2/Sum_grad/ConstConst*
valueB:??*
_output_shapes
:*
dtype0
?
gradients_2/Sum_grad/TileTilegradients_2/Sum_grad/Reshapegradients_2/Sum_grad/Const*

Tmultiples0*
_output_shapes

:??*
T0
r
gradients_2/mul_3_grad/MulMulgradients_2/Sum_grad/TilePlaceholder_9*
_output_shapes

:??*
T0
o
gradients_2/mul_3_grad/Mul_1Mulgradients_2/Sum_grad/Tileconcat_1*
T0*
_output_shapes

:??
`
gradients_2/concat_1_grad/RankConst*
dtype0*
_output_shapes
: *
value	B :
y
gradients_2/concat_1_grad/modFloorModconcat_1/axisgradients_2/concat_1_grad/Rank*
T0*
_output_shapes
: 
j
gradients_2/concat_1_grad/ShapeConst*
_output_shapes
:*
valueB:?x*
dtype0
l
!gradients_2/concat_1_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:?
m
!gradients_2/concat_1_grad/Shape_2Const*
valueB:??*
dtype0*
_output_shapes
:
l
!gradients_2/concat_1_grad/Shape_3Const*
valueB:?*
_output_shapes
:*
dtype0
l
!gradients_2/concat_1_grad/Shape_4Const*
_output_shapes
:*
valueB:?*
dtype0
k
!gradients_2/concat_1_grad/Shape_5Const*
dtype0*
_output_shapes
:*
valueB:
k
!gradients_2/concat_1_grad/Shape_6Const*
dtype0*
_output_shapes
:*
valueB:
?
&gradients_2/concat_1_grad/ConcatOffsetConcatOffsetgradients_2/concat_1_grad/modgradients_2/concat_1_grad/Shape!gradients_2/concat_1_grad/Shape_1!gradients_2/concat_1_grad/Shape_2!gradients_2/concat_1_grad/Shape_3!gradients_2/concat_1_grad/Shape_4!gradients_2/concat_1_grad/Shape_5!gradients_2/concat_1_grad/Shape_6*
N*>
_output_shapes,
*:::::::
?
gradients_2/concat_1_grad/SliceSlicegradients_2/mul_3_grad/Mul&gradients_2/concat_1_grad/ConcatOffsetgradients_2/concat_1_grad/Shape*
T0*
_output_shapes	
:?x*
Index0
?
!gradients_2/concat_1_grad/Slice_1Slicegradients_2/mul_3_grad/Mul(gradients_2/concat_1_grad/ConcatOffset:1!gradients_2/concat_1_grad/Shape_1*
_output_shapes	
:?*
T0*
Index0
?
!gradients_2/concat_1_grad/Slice_2Slicegradients_2/mul_3_grad/Mul(gradients_2/concat_1_grad/ConcatOffset:2!gradients_2/concat_1_grad/Shape_2*
Index0*
_output_shapes

:??*
T0
?
!gradients_2/concat_1_grad/Slice_3Slicegradients_2/mul_3_grad/Mul(gradients_2/concat_1_grad/ConcatOffset:3!gradients_2/concat_1_grad/Shape_3*
T0*
Index0*
_output_shapes	
:?
?
!gradients_2/concat_1_grad/Slice_4Slicegradients_2/mul_3_grad/Mul(gradients_2/concat_1_grad/ConcatOffset:4!gradients_2/concat_1_grad/Shape_4*
T0*
_output_shapes	
:?*
Index0
?
!gradients_2/concat_1_grad/Slice_5Slicegradients_2/mul_3_grad/Mul(gradients_2/concat_1_grad/ConcatOffset:5!gradients_2/concat_1_grad/Shape_5*
T0*
_output_shapes
:*
Index0
?
!gradients_2/concat_1_grad/Slice_6Slicegradients_2/mul_3_grad/Mul(gradients_2/concat_1_grad/ConcatOffset:6!gradients_2/concat_1_grad/Shape_6*
T0*
Index0*
_output_shapes
:
q
 gradients_2/Reshape_7_grad/ShapeConst*
valueB"<      *
_output_shapes
:*
dtype0
?
"gradients_2/Reshape_7_grad/ReshapeReshapegradients_2/concat_1_grad/Slice gradients_2/Reshape_7_grad/Shape*
Tshape0*
T0*
_output_shapes
:	<?
k
 gradients_2/Reshape_8_grad/ShapeConst*
_output_shapes
:*
valueB:?*
dtype0
?
"gradients_2/Reshape_8_grad/ReshapeReshape!gradients_2/concat_1_grad/Slice_1 gradients_2/Reshape_8_grad/Shape*
_output_shapes	
:?*
T0*
Tshape0
q
 gradients_2/Reshape_9_grad/ShapeConst*
dtype0*
valueB"      *
_output_shapes
:
?
"gradients_2/Reshape_9_grad/ReshapeReshape!gradients_2/concat_1_grad/Slice_2 gradients_2/Reshape_9_grad/Shape* 
_output_shapes
:
??*
T0*
Tshape0
l
!gradients_2/Reshape_10_grad/ShapeConst*
_output_shapes
:*
valueB:?*
dtype0
?
#gradients_2/Reshape_10_grad/ReshapeReshape!gradients_2/concat_1_grad/Slice_3!gradients_2/Reshape_10_grad/Shape*
T0*
_output_shapes	
:?*
Tshape0
r
!gradients_2/Reshape_11_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB"      
?
#gradients_2/Reshape_11_grad/ReshapeReshape!gradients_2/concat_1_grad/Slice_4!gradients_2/Reshape_11_grad/Shape*
T0*
_output_shapes
:	?*
Tshape0
k
!gradients_2/Reshape_12_grad/ShapeConst*
dtype0*
valueB:*
_output_shapes
:
?
#gradients_2/Reshape_12_grad/ReshapeReshape!gradients_2/concat_1_grad/Slice_5!gradients_2/Reshape_12_grad/Shape*
Tshape0*
_output_shapes
:*
T0
k
!gradients_2/Reshape_13_grad/ShapeConst*
_output_shapes
:*
dtype0*
valueB:
?
#gradients_2/Reshape_13_grad/ReshapeReshape!gradients_2/concat_1_grad/Slice_6!gradients_2/Reshape_13_grad/Shape*
T0*
_output_shapes
:*
Tshape0
?
Agradients_2/gradients_1/pi/dense/MatMul_grad/MatMul_1_grad/MatMulMatMul'gradients_1/pi/dense/Tanh_grad/TanhGrad"gradients_2/Reshape_7_grad/Reshape*
transpose_a( *'
_output_shapes
:?????????<*
T0*
transpose_b(
?
Cgradients_2/gradients_1/pi/dense/MatMul_grad/MatMul_1_grad/MatMul_1MatMulPlaceholder"gradients_2/Reshape_7_grad/Reshape*(
_output_shapes
:??????????*
transpose_b( *
T0*
transpose_a( 
?
Dgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/ShapeShape'gradients_1/pi/dense/Tanh_grad/TanhGrad*
out_type0*
_output_shapes
:*
T0
?
Cgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/RankConst*
value	B :*
dtype0*
_output_shapes
: 
?
Fgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/Shape_1Const*
_output_shapes
:*
valueB:?*
dtype0
?
Rgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
?
Tgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/strided_slice/stack_1Const*
_output_shapes
:*
valueB:
?????????*
dtype0
?
Tgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/strided_slice/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
?
Lgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/strided_sliceStridedSliceDgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/ShapeRgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/strided_slice/stackTgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/strided_slice/stack_1Tgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/strided_slice/stack_2*
new_axis_mask *

begin_mask*
end_mask *
_output_shapes
:*
Index0*
ellipsis_mask *
shrink_axis_mask *
T0
?
Ngradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/ones_like/ShapeConst*
dtype0*
_output_shapes
:*
valueB:
?
Ngradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/ones_like/ConstConst*
dtype0*
_output_shapes
: *
value	B :
?
Hgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/ones_likeFillNgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/ones_like/ShapeNgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/ones_like/Const*
_output_shapes
:*

index_type0*
T0
?
Jgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : 
?
Egradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/concatConcatV2Hgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/ones_likeFgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/Shape_1Jgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/concat/axis*
T0*
_output_shapes
:*
N*

Tidx0
?
Tgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/strided_slice_1/stackConst*
dtype0*
valueB: *
_output_shapes
:
?
Vgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/strided_slice_1/stack_1Const*
valueB:
?????????*
dtype0*
_output_shapes
:
?
Vgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/strided_slice_1/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
?
Ngradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/strided_slice_1StridedSliceDgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/ShapeTgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/strided_slice_1/stackVgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/strided_slice_1/stack_1Vgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/strided_slice_1/stack_2*
new_axis_mask *
_output_shapes
:*

begin_mask*
end_mask *
Index0*
shrink_axis_mask *
ellipsis_mask *
T0
?
Pgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/concat_1/values_1Const*
valueB:*
dtype0*
_output_shapes
:
?
Lgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
Ggradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/concat_1ConcatV2Ngradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/strided_slice_1Pgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/concat_1/values_1Lgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/concat_1/axis*

Tidx0*
T0*
_output_shapes
:*
N
?
Fgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/ReshapeReshape"gradients_2/Reshape_8_grad/ReshapeEgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/concat*
T0*
Tshape0*
_output_shapes
:	?
?
Cgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/TileTileFgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/ReshapeGgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/concat_1*(
_output_shapes
:??????????*

Tmultiples0*
T0
?
Cgradients_2/gradients_1/pi/dense_1/MatMul_grad/MatMul_1_grad/MatMulMatMul)gradients_1/pi/dense_1/Tanh_grad/TanhGrad"gradients_2/Reshape_9_grad/Reshape*
transpose_b(*(
_output_shapes
:??????????*
T0*
transpose_a( 
?
Egradients_2/gradients_1/pi/dense_1/MatMul_grad/MatMul_1_grad/MatMul_1MatMulpi/dense/Tanh"gradients_2/Reshape_9_grad/Reshape*
transpose_b( *
transpose_a( *
T0*(
_output_shapes
:??????????
?
Fgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/ShapeShape)gradients_1/pi/dense_1/Tanh_grad/TanhGrad*
out_type0*
T0*
_output_shapes
:
?
Egradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/RankConst*
_output_shapes
: *
dtype0*
value	B :
?
Hgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/Shape_1Const*
valueB:?*
_output_shapes
:*
dtype0
?
Tgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/strided_slice/stackConst*
dtype0*
valueB: *
_output_shapes
:
?
Vgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/strided_slice/stack_1Const*
valueB:
?????????*
_output_shapes
:*
dtype0
?
Vgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
?
Ngradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/strided_sliceStridedSliceFgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/ShapeTgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/strided_slice/stackVgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/strided_slice/stack_1Vgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/strided_slice/stack_2*
new_axis_mask *
_output_shapes
:*
ellipsis_mask *
end_mask *
Index0*
T0*

begin_mask*
shrink_axis_mask 
?
Pgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/ones_like/ShapeConst*
dtype0*
valueB:*
_output_shapes
:
?
Pgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/ones_like/ConstConst*
_output_shapes
: *
value	B :*
dtype0
?
Jgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/ones_likeFillPgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/ones_like/ShapePgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/ones_like/Const*

index_type0*
T0*
_output_shapes
:
?
Lgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : 
?
Ggradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/concatConcatV2Jgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/ones_likeHgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/Shape_1Lgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/concat/axis*

Tidx0*
N*
_output_shapes
:*
T0
?
Vgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:
?
Xgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/strided_slice_1/stack_1Const*
valueB:
?????????*
_output_shapes
:*
dtype0
?
Xgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/strided_slice_1/stack_2Const*
dtype0*
valueB:*
_output_shapes
:
?
Pgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/strided_slice_1StridedSliceFgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/ShapeVgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/strided_slice_1/stackXgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/strided_slice_1/stack_1Xgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/strided_slice_1/stack_2*
end_mask *
new_axis_mask *

begin_mask*
T0*
ellipsis_mask *
_output_shapes
:*
shrink_axis_mask *
Index0
?
Rgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/concat_1/values_1Const*
dtype0*
_output_shapes
:*
valueB:
?
Ngradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/concat_1/axisConst*
_output_shapes
: *
dtype0*
value	B : 
?
Igradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/concat_1ConcatV2Pgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/strided_slice_1Rgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/concat_1/values_1Ngradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/concat_1/axis*
T0*

Tidx0*
_output_shapes
:*
N
?
Hgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/ReshapeReshape#gradients_2/Reshape_10_grad/ReshapeGgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/concat*
Tshape0*
_output_shapes
:	?*
T0
?
Egradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/TileTileHgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/ReshapeIgradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/concat_1*(
_output_shapes
:??????????*

Tmultiples0*
T0
?
Cgradients_2/gradients_1/pi/dense_2/MatMul_grad/MatMul_1_grad/MatMulMatMul#gradients_1/pi/sub_2_grad/Reshape_1#gradients_2/Reshape_11_grad/Reshape*
transpose_b(*
T0*
transpose_a( *(
_output_shapes
:??????????
?
Egradients_2/gradients_1/pi/dense_2/MatMul_grad/MatMul_1_grad/MatMul_1MatMulpi/dense_1/Tanh#gradients_2/Reshape_11_grad/Reshape*
transpose_b( *'
_output_shapes
:?????????*
transpose_a( *
T0
?
Fgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/ShapeShape#gradients_1/pi/sub_2_grad/Reshape_1*
T0*
_output_shapes
:*
out_type0
?
Egradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/RankConst*
_output_shapes
: *
value	B :*
dtype0
?
Hgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/Shape_1Const*
valueB:*
_output_shapes
:*
dtype0
?
Tgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/strided_slice/stackConst*
_output_shapes
:*
valueB: *
dtype0
?
Vgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
?????????
?
Vgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/strided_slice/stack_2Const*
_output_shapes
:*
valueB:*
dtype0
?
Ngradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/strided_sliceStridedSliceFgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/ShapeTgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/strided_slice/stackVgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/strided_slice/stack_1Vgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/strided_slice/stack_2*
Index0*
shrink_axis_mask *
new_axis_mask *
ellipsis_mask *
_output_shapes
:*

begin_mask*
end_mask *
T0
?
Pgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/ones_like/ShapeConst*
valueB:*
dtype0*
_output_shapes
:
?
Pgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/ones_like/ConstConst*
_output_shapes
: *
dtype0*
value	B :
?
Jgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/ones_likeFillPgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/ones_like/ShapePgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/ones_like/Const*
_output_shapes
:*

index_type0*
T0
?
Lgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/concat/axisConst*
dtype0*
value	B : *
_output_shapes
: 
?
Ggradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/concatConcatV2Jgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/ones_likeHgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/Shape_1Lgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/concat/axis*
N*
_output_shapes
:*

Tidx0*
T0
?
Vgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/strided_slice_1/stackConst*
valueB: *
_output_shapes
:*
dtype0
?
Xgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/strided_slice_1/stack_1Const*
valueB:
?????????*
_output_shapes
:*
dtype0
?
Xgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/strided_slice_1/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
?
Pgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/strided_slice_1StridedSliceFgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/ShapeVgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/strided_slice_1/stackXgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/strided_slice_1/stack_1Xgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/strided_slice_1/stack_2*
_output_shapes
:*

begin_mask*
end_mask *
shrink_axis_mask *
Index0*
new_axis_mask *
T0*
ellipsis_mask 
?
Rgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/concat_1/values_1Const*
_output_shapes
:*
dtype0*
valueB:
?
Ngradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
Igradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/concat_1ConcatV2Pgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/strided_slice_1Rgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/concat_1/values_1Ngradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/concat_1/axis*
N*

Tidx0*
T0*
_output_shapes
:
?
Hgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/ReshapeReshape#gradients_2/Reshape_12_grad/ReshapeGgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/concat*
_output_shapes

:*
T0*
Tshape0
?
Egradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/TileTileHgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/ReshapeIgradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/concat_1*

Tmultiples0*
T0*'
_output_shapes
:?????????
?
gradients_2/AddNAddNCgradients_2/gradients_1/pi/dense/MatMul_grad/MatMul_1_grad/MatMul_1Cgradients_2/gradients_1/pi/dense/BiasAdd_grad/BiasAddGrad_grad/Tile*(
_output_shapes
:??????????*
N*V
_classL
JHloc:@gradients_2/gradients_1/pi/dense/MatMul_grad/MatMul_1_grad/MatMul_1*
T0
?
>gradients_2/gradients_1/pi/dense/Tanh_grad/TanhGrad_grad/mul/yConst^gradients_2/AddN*
dtype0*
_output_shapes
: *
valueB
 *   ?
?
<gradients_2/gradients_1/pi/dense/Tanh_grad/TanhGrad_grad/mulMulgradients_2/AddN>gradients_2/gradients_1/pi/dense/Tanh_grad/TanhGrad_grad/mul/y*
T0*(
_output_shapes
:??????????
?
>gradients_2/gradients_1/pi/dense/Tanh_grad/TanhGrad_grad/mul_1Mul<gradients_2/gradients_1/pi/dense/Tanh_grad/TanhGrad_grad/mul)gradients_1/pi/dense_1/MatMul_grad/MatMul*(
_output_shapes
:??????????*
T0
?
>gradients_2/gradients_1/pi/dense/Tanh_grad/TanhGrad_grad/mul_2Mul>gradients_2/gradients_1/pi/dense/Tanh_grad/TanhGrad_grad/mul_1pi/dense/Tanh*(
_output_shapes
:??????????*
T0
?
Agradients_2/gradients_1/pi/dense/Tanh_grad/TanhGrad_grad/TanhGradTanhGradpi/dense/Tanhgradients_2/AddN*
T0*(
_output_shapes
:??????????
?
:gradients_2/gradients_1/pi/mul_5_grad/Reshape_1_grad/ShapeConst*
valueB:*
_output_shapes
:*
dtype0
?
<gradients_2/gradients_1/pi/mul_5_grad/Reshape_1_grad/ReshapeReshape#gradients_2/Reshape_13_grad/Reshape:gradients_2/gradients_1/pi/mul_5_grad/Reshape_1_grad/Shape*
_output_shapes
:*
T0*
Tshape0
?
Agradients_2/gradients_1/pi/dense_1/MatMul_grad/MatMul_grad/MatMulMatMulAgradients_2/gradients_1/pi/dense/Tanh_grad/TanhGrad_grad/TanhGradpi/dense_1/kernel/read*(
_output_shapes
:??????????*
T0*
transpose_b( *
transpose_a( 
?
Cgradients_2/gradients_1/pi/dense_1/MatMul_grad/MatMul_grad/MatMul_1MatMulAgradients_2/gradients_1/pi/dense/Tanh_grad/TanhGrad_grad/TanhGrad)gradients_1/pi/dense_1/Tanh_grad/TanhGrad*
T0* 
_output_shapes
:
??*
transpose_b( *
transpose_a(
?
6gradients_2/gradients_1/pi/mul_5_grad/Sum_1_grad/ShapeConst*
dtype0*
valueB:*
_output_shapes
:
?
5gradients_2/gradients_1/pi/mul_5_grad/Sum_1_grad/SizeConst*
value	B :*
dtype0*I
_class?
=;loc:@gradients_2/gradients_1/pi/mul_5_grad/Sum_1_grad/Shape*
_output_shapes
: 
?
4gradients_2/gradients_1/pi/mul_5_grad/Sum_1_grad/addAdd1gradients_1/pi/mul_5_grad/BroadcastGradientArgs:15gradients_2/gradients_1/pi/mul_5_grad/Sum_1_grad/Size*
T0*#
_output_shapes
:?????????*I
_class?
=;loc:@gradients_2/gradients_1/pi/mul_5_grad/Sum_1_grad/Shape
?
4gradients_2/gradients_1/pi/mul_5_grad/Sum_1_grad/modFloorMod4gradients_2/gradients_1/pi/mul_5_grad/Sum_1_grad/add5gradients_2/gradients_1/pi/mul_5_grad/Sum_1_grad/Size*#
_output_shapes
:?????????*I
_class?
=;loc:@gradients_2/gradients_1/pi/mul_5_grad/Sum_1_grad/Shape*
T0
?
8gradients_2/gradients_1/pi/mul_5_grad/Sum_1_grad/Shape_1Shape4gradients_2/gradients_1/pi/mul_5_grad/Sum_1_grad/mod*
T0*
out_type0*I
_class?
=;loc:@gradients_2/gradients_1/pi/mul_5_grad/Sum_1_grad/Shape*
_output_shapes
:
?
<gradients_2/gradients_1/pi/mul_5_grad/Sum_1_grad/range/startConst*
dtype0*
_output_shapes
: *
value	B : *I
_class?
=;loc:@gradients_2/gradients_1/pi/mul_5_grad/Sum_1_grad/Shape
?
<gradients_2/gradients_1/pi/mul_5_grad/Sum_1_grad/range/deltaConst*
dtype0*I
_class?
=;loc:@gradients_2/gradients_1/pi/mul_5_grad/Sum_1_grad/Shape*
value	B :*
_output_shapes
: 
?
6gradients_2/gradients_1/pi/mul_5_grad/Sum_1_grad/rangeRange<gradients_2/gradients_1/pi/mul_5_grad/Sum_1_grad/range/start5gradients_2/gradients_1/pi/mul_5_grad/Sum_1_grad/Size<gradients_2/gradients_1/pi/mul_5_grad/Sum_1_grad/range/delta*

Tidx0*I
_class?
=;loc:@gradients_2/gradients_1/pi/mul_5_grad/Sum_1_grad/Shape*
_output_shapes
:
?
;gradients_2/gradients_1/pi/mul_5_grad/Sum_1_grad/Fill/valueConst*I
_class?
=;loc:@gradients_2/gradients_1/pi/mul_5_grad/Sum_1_grad/Shape*
_output_shapes
: *
dtype0*
value	B :
?
5gradients_2/gradients_1/pi/mul_5_grad/Sum_1_grad/FillFill8gradients_2/gradients_1/pi/mul_5_grad/Sum_1_grad/Shape_1;gradients_2/gradients_1/pi/mul_5_grad/Sum_1_grad/Fill/value*#
_output_shapes
:?????????*I
_class?
=;loc:@gradients_2/gradients_1/pi/mul_5_grad/Sum_1_grad/Shape*

index_type0*
T0
?
>gradients_2/gradients_1/pi/mul_5_grad/Sum_1_grad/DynamicStitchDynamicStitch6gradients_2/gradients_1/pi/mul_5_grad/Sum_1_grad/range4gradients_2/gradients_1/pi/mul_5_grad/Sum_1_grad/mod6gradients_2/gradients_1/pi/mul_5_grad/Sum_1_grad/Shape5gradients_2/gradients_1/pi/mul_5_grad/Sum_1_grad/Fill*
T0*
N*
_output_shapes
:*I
_class?
=;loc:@gradients_2/gradients_1/pi/mul_5_grad/Sum_1_grad/Shape
?
:gradients_2/gradients_1/pi/mul_5_grad/Sum_1_grad/Maximum/yConst*
value	B :*
dtype0*I
_class?
=;loc:@gradients_2/gradients_1/pi/mul_5_grad/Sum_1_grad/Shape*
_output_shapes
: 
?
8gradients_2/gradients_1/pi/mul_5_grad/Sum_1_grad/MaximumMaximum>gradients_2/gradients_1/pi/mul_5_grad/Sum_1_grad/DynamicStitch:gradients_2/gradients_1/pi/mul_5_grad/Sum_1_grad/Maximum/y*
_output_shapes
:*
T0*I
_class?
=;loc:@gradients_2/gradients_1/pi/mul_5_grad/Sum_1_grad/Shape
?
9gradients_2/gradients_1/pi/mul_5_grad/Sum_1_grad/floordivFloorDiv6gradients_2/gradients_1/pi/mul_5_grad/Sum_1_grad/Shape8gradients_2/gradients_1/pi/mul_5_grad/Sum_1_grad/Maximum*
_output_shapes
:*
T0*I
_class?
=;loc:@gradients_2/gradients_1/pi/mul_5_grad/Sum_1_grad/Shape
?
8gradients_2/gradients_1/pi/mul_5_grad/Sum_1_grad/ReshapeReshape<gradients_2/gradients_1/pi/mul_5_grad/Reshape_1_grad/Reshape>gradients_2/gradients_1/pi/mul_5_grad/Sum_1_grad/DynamicStitch*
_output_shapes
:*
Tshape0*
T0
?
5gradients_2/gradients_1/pi/mul_5_grad/Sum_1_grad/TileTile8gradients_2/gradients_1/pi/mul_5_grad/Sum_1_grad/Reshape9gradients_2/gradients_1/pi/mul_5_grad/Sum_1_grad/floordiv*
_output_shapes
:*
T0*

Tmultiples0
?
gradients_2/AddN_1AddNEgradients_2/gradients_1/pi/dense_1/MatMul_grad/MatMul_1_grad/MatMul_1Egradients_2/gradients_1/pi/dense_1/BiasAdd_grad/BiasAddGrad_grad/TileAgradients_2/gradients_1/pi/dense_1/MatMul_grad/MatMul_grad/MatMul*(
_output_shapes
:??????????*X
_classN
LJloc:@gradients_2/gradients_1/pi/dense_1/MatMul_grad/MatMul_1_grad/MatMul_1*
T0*
N
?
@gradients_2/gradients_1/pi/dense_1/Tanh_grad/TanhGrad_grad/mul/yConst^gradients_2/AddN_1*
dtype0*
_output_shapes
: *
valueB
 *   ?
?
>gradients_2/gradients_1/pi/dense_1/Tanh_grad/TanhGrad_grad/mulMulgradients_2/AddN_1@gradients_2/gradients_1/pi/dense_1/Tanh_grad/TanhGrad_grad/mul/y*
T0*(
_output_shapes
:??????????
?
@gradients_2/gradients_1/pi/dense_1/Tanh_grad/TanhGrad_grad/mul_1Mul>gradients_2/gradients_1/pi/dense_1/Tanh_grad/TanhGrad_grad/mul)gradients_1/pi/dense_2/MatMul_grad/MatMul*(
_output_shapes
:??????????*
T0
?
@gradients_2/gradients_1/pi/dense_1/Tanh_grad/TanhGrad_grad/mul_2Mul@gradients_2/gradients_1/pi/dense_1/Tanh_grad/TanhGrad_grad/mul_1pi/dense_1/Tanh*(
_output_shapes
:??????????*
T0
?
Cgradients_2/gradients_1/pi/dense_1/Tanh_grad/TanhGrad_grad/TanhGradTanhGradpi/dense_1/Tanhgradients_2/AddN_1*
T0*(
_output_shapes
:??????????
y
6gradients_2/gradients_1/pi/mul_5_grad/Mul_1_grad/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
?
8gradients_2/gradients_1/pi/mul_5_grad/Mul_1_grad/Shape_1Const*
valueB:*
_output_shapes
:*
dtype0
?
Fgradients_2/gradients_1/pi/mul_5_grad/Mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs6gradients_2/gradients_1/pi/mul_5_grad/Mul_1_grad/Shape8gradients_2/gradients_1/pi/mul_5_grad/Mul_1_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
4gradients_2/gradients_1/pi/mul_5_grad/Mul_1_grad/MulMul5gradients_2/gradients_1/pi/mul_5_grad/Sum_1_grad/Tilegradients_1/pi/Exp_3_grad/mul*
T0*
_output_shapes
:
?
4gradients_2/gradients_1/pi/mul_5_grad/Mul_1_grad/SumSum4gradients_2/gradients_1/pi/mul_5_grad/Mul_1_grad/MulFgradients_2/gradients_1/pi/mul_5_grad/Mul_1_grad/BroadcastGradientArgs*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
?
8gradients_2/gradients_1/pi/mul_5_grad/Mul_1_grad/ReshapeReshape4gradients_2/gradients_1/pi/mul_5_grad/Mul_1_grad/Sum6gradients_2/gradients_1/pi/mul_5_grad/Mul_1_grad/Shape*
_output_shapes
: *
T0*
Tshape0
?
6gradients_2/gradients_1/pi/mul_5_grad/Mul_1_grad/Mul_1Mul
pi/mul_5/x5gradients_2/gradients_1/pi/mul_5_grad/Sum_1_grad/Tile*
_output_shapes
:*
T0
?
6gradients_2/gradients_1/pi/mul_5_grad/Mul_1_grad/Sum_1Sum6gradients_2/gradients_1/pi/mul_5_grad/Mul_1_grad/Mul_1Hgradients_2/gradients_1/pi/mul_5_grad/Mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
T0*
	keep_dims( 
?
:gradients_2/gradients_1/pi/mul_5_grad/Mul_1_grad/Reshape_1Reshape6gradients_2/gradients_1/pi/mul_5_grad/Mul_1_grad/Sum_18gradients_2/gradients_1/pi/mul_5_grad/Mul_1_grad/Shape_1*
_output_shapes
:*
Tshape0*
T0
?
Agradients_2/gradients_1/pi/dense_2/MatMul_grad/MatMul_grad/MatMulMatMulCgradients_2/gradients_1/pi/dense_1/Tanh_grad/TanhGrad_grad/TanhGradpi/dense_2/kernel/read*
transpose_a( *'
_output_shapes
:?????????*
T0*
transpose_b( 
?
Cgradients_2/gradients_1/pi/dense_2/MatMul_grad/MatMul_grad/MatMul_1MatMulCgradients_2/gradients_1/pi/dense_1/Tanh_grad/TanhGrad_grad/TanhGrad#gradients_1/pi/sub_2_grad/Reshape_1*
T0*
transpose_b( *
_output_shapes
:	?*
transpose_a(
?
2gradients_2/gradients_1/pi/Exp_3_grad/mul_grad/MulMul:gradients_2/gradients_1/pi/mul_5_grad/Mul_1_grad/Reshape_1pi/Exp_3*
_output_shapes
:*
T0
?
4gradients_2/gradients_1/pi/Exp_3_grad/mul_grad/Mul_1Mul:gradients_2/gradients_1/pi/mul_5_grad/Mul_1_grad/Reshape_1#gradients_1/pi/add_7_grad/Reshape_1*
_output_shapes
:*
T0
?
gradients_2/AddN_2AddNEgradients_2/gradients_1/pi/dense_2/MatMul_grad/MatMul_1_grad/MatMul_1Egradients_2/gradients_1/pi/dense_2/BiasAdd_grad/BiasAddGrad_grad/TileAgradients_2/gradients_1/pi/dense_2/MatMul_grad/MatMul_grad/MatMul*
T0*'
_output_shapes
:?????????*X
_classN
LJloc:@gradients_2/gradients_1/pi/dense_2/MatMul_grad/MatMul_1_grad/MatMul_1*
N
?
:gradients_2/gradients_1/pi/sub_2_grad/Reshape_1_grad/ShapeShapegradients_1/pi/sub_2_grad/Neg*
T0*#
_output_shapes
:?????????*
out_type0
?
<gradients_2/gradients_1/pi/sub_2_grad/Reshape_1_grad/ReshapeReshapegradients_2/AddN_2:gradients_2/gradients_1/pi/sub_2_grad/Reshape_1_grad/Shape*
T0*
_output_shapes
:*
Tshape0
?
gradients_2/pi/Exp_3_grad/mulMul4gradients_2/gradients_1/pi/Exp_3_grad/mul_grad/Mul_1pi/Exp_3*
_output_shapes
:*
T0
?
2gradients_2/gradients_1/pi/sub_2_grad/Neg_grad/NegNeg<gradients_2/gradients_1/pi/sub_2_grad/Reshape_1_grad/Reshape*
_output_shapes
:*
T0
b
gradients_2/pi/mul_5_grad/ShapeConst*
_output_shapes
: *
dtype0*
valueB 
k
!gradients_2/pi/mul_5_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:
?
/gradients_2/pi/mul_5_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_2/pi/mul_5_grad/Shape!gradients_2/pi/mul_5_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
y
gradients_2/pi/mul_5_grad/MulMulgradients_2/pi/Exp_3_grad/mulpi/log_std/read*
_output_shapes
:*
T0
?
gradients_2/pi/mul_5_grad/SumSumgradients_2/pi/mul_5_grad/Mul/gradients_2/pi/mul_5_grad/BroadcastGradientArgs*
T0*
	keep_dims( *
_output_shapes
: *

Tidx0
?
!gradients_2/pi/mul_5_grad/ReshapeReshapegradients_2/pi/mul_5_grad/Sumgradients_2/pi/mul_5_grad/Shape*
Tshape0*
_output_shapes
: *
T0
v
gradients_2/pi/mul_5_grad/Mul_1Mul
pi/mul_5/xgradients_2/pi/Exp_3_grad/mul*
_output_shapes
:*
T0
?
gradients_2/pi/mul_5_grad/Sum_1Sumgradients_2/pi/mul_5_grad/Mul_11gradients_2/pi/mul_5_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
?
#gradients_2/pi/mul_5_grad/Reshape_1Reshapegradients_2/pi/mul_5_grad/Sum_1!gradients_2/pi/mul_5_grad/Shape_1*
T0*
_output_shapes
:*
Tshape0
?
6gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/ShapeShape!gradients_1/pi/pow_2_grad/Reshape*
T0*
_output_shapes
:*
out_type0
?
5gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/SizeConst*I
_class?
=;loc:@gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
?
4gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/addAdd1gradients_1/pi/sub_2_grad/BroadcastGradientArgs:15gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/Size*
T0*I
_class?
=;loc:@gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/Shape*#
_output_shapes
:?????????
?
4gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/modFloorMod4gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/add5gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/Size*I
_class?
=;loc:@gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/Shape*
T0*#
_output_shapes
:?????????
?
8gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/Shape_1Shape4gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/mod*
out_type0*
_output_shapes
:*I
_class?
=;loc:@gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/Shape*
T0
?
<gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/range/startConst*I
_class?
=;loc:@gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/Shape*
value	B : *
dtype0*
_output_shapes
: 
?
<gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/range/deltaConst*I
_class?
=;loc:@gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/Shape*
dtype0*
value	B :*
_output_shapes
: 
?
6gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/rangeRange<gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/range/start5gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/Size<gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/range/delta*I
_class?
=;loc:@gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/Shape*
_output_shapes
:*

Tidx0
?
;gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/Fill/valueConst*
_output_shapes
: *I
_class?
=;loc:@gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/Shape*
dtype0*
value	B :
?
5gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/FillFill8gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/Shape_1;gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/Fill/value*I
_class?
=;loc:@gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/Shape*

index_type0*
T0*#
_output_shapes
:?????????
?
>gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/DynamicStitchDynamicStitch6gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/range4gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/mod6gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/Shape5gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/Fill*
T0*#
_output_shapes
:?????????*
N*I
_class?
=;loc:@gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/Shape
?
:gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/Maximum/yConst*
value	B :*
_output_shapes
: *
dtype0*I
_class?
=;loc:@gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/Shape
?
8gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/MaximumMaximum>gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/DynamicStitch:gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/Maximum/y*
T0*#
_output_shapes
:?????????*I
_class?
=;loc:@gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/Shape
?
9gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/floordivFloorDiv6gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/Shape8gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/Maximum*
_output_shapes
:*I
_class?
=;loc:@gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/Shape*
T0
?
8gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/ReshapeReshape2gradients_2/gradients_1/pi/sub_2_grad/Neg_grad/Neg>gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/DynamicStitch*
_output_shapes
:*
T0*
Tshape0
?
5gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/TileTile8gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/Reshape9gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/floordiv*

Tmultiples0*'
_output_shapes
:?????????*
T0
?
8gradients_2/gradients_1/pi/pow_2_grad/Reshape_grad/ShapeShapegradients_1/pi/pow_2_grad/Sum*
T0*#
_output_shapes
:?????????*
out_type0
?
:gradients_2/gradients_1/pi/pow_2_grad/Reshape_grad/ReshapeReshape5gradients_2/gradients_1/pi/sub_2_grad/Sum_1_grad/Tile8gradients_2/gradients_1/pi/pow_2_grad/Reshape_grad/Shape*
T0*
_output_shapes
:*
Tshape0
?
4gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/ShapeShapegradients_1/pi/pow_2_grad/mul_1*
T0*
_output_shapes
:*
out_type0
?
3gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/SizeConst*G
_class=
;9loc:@gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/Shape*
value	B :*
dtype0*
_output_shapes
: 
?
2gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/addAdd/gradients_1/pi/pow_2_grad/BroadcastGradientArgs3gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/Size*
T0*G
_class=
;9loc:@gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/Shape*#
_output_shapes
:?????????
?
2gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/modFloorMod2gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/add3gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/Size*
T0*#
_output_shapes
:?????????*G
_class=
;9loc:@gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/Shape
?
6gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/Shape_1Shape2gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/mod*
_output_shapes
:*
T0*
out_type0*G
_class=
;9loc:@gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/Shape
?
:gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/range/startConst*
value	B : *
_output_shapes
: *G
_class=
;9loc:@gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/Shape*
dtype0
?
:gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/range/deltaConst*G
_class=
;9loc:@gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/Shape*
dtype0*
value	B :*
_output_shapes
: 
?
4gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/rangeRange:gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/range/start3gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/Size:gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/range/delta*
_output_shapes
:*G
_class=
;9loc:@gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/Shape*

Tidx0
?
9gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/Fill/valueConst*
dtype0*G
_class=
;9loc:@gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/Shape*
_output_shapes
: *
value	B :
?
3gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/FillFill6gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/Shape_19gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/Fill/value*
T0*G
_class=
;9loc:@gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/Shape*

index_type0*#
_output_shapes
:?????????
?
<gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/DynamicStitchDynamicStitch4gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/range2gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/mod4gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/Shape3gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/Fill*
T0*#
_output_shapes
:?????????*G
_class=
;9loc:@gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/Shape*
N
?
8gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/Maximum/yConst*
dtype0*
value	B :*G
_class=
;9loc:@gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/Shape*
_output_shapes
: 
?
6gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/MaximumMaximum<gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/DynamicStitch8gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/Maximum/y*G
_class=
;9loc:@gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/Shape*#
_output_shapes
:?????????*
T0
?
7gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/floordivFloorDiv4gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/Shape6gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/Maximum*G
_class=
;9loc:@gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/Shape*
T0*
_output_shapes
:
?
6gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/ReshapeReshape:gradients_2/gradients_1/pi/pow_2_grad/Reshape_grad/Reshape<gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/DynamicStitch*
_output_shapes
:*
T0*
Tshape0
?
3gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/TileTile6gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/Reshape7gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/floordiv*
T0*

Tmultiples0*'
_output_shapes
:?????????
?
6gradients_2/gradients_1/pi/pow_2_grad/mul_1_grad/ShapeShapegradients_1/pi/pow_2_grad/mul*
_output_shapes
:*
T0*
out_type0
?
8gradients_2/gradients_1/pi/pow_2_grad/mul_1_grad/Shape_1Shapegradients_1/pi/pow_2_grad/Pow*
T0*
_output_shapes
:*
out_type0
?
Fgradients_2/gradients_1/pi/pow_2_grad/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgs6gradients_2/gradients_1/pi/pow_2_grad/mul_1_grad/Shape8gradients_2/gradients_1/pi/pow_2_grad/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
4gradients_2/gradients_1/pi/pow_2_grad/mul_1_grad/MulMul3gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/Tilegradients_1/pi/pow_2_grad/Pow*'
_output_shapes
:?????????*
T0
?
4gradients_2/gradients_1/pi/pow_2_grad/mul_1_grad/SumSum4gradients_2/gradients_1/pi/pow_2_grad/mul_1_grad/MulFgradients_2/gradients_1/pi/pow_2_grad/mul_1_grad/BroadcastGradientArgs*
T0*
	keep_dims( *

Tidx0*
_output_shapes
:
?
8gradients_2/gradients_1/pi/pow_2_grad/mul_1_grad/ReshapeReshape4gradients_2/gradients_1/pi/pow_2_grad/mul_1_grad/Sum6gradients_2/gradients_1/pi/pow_2_grad/mul_1_grad/Shape*'
_output_shapes
:?????????*
T0*
Tshape0
?
6gradients_2/gradients_1/pi/pow_2_grad/mul_1_grad/Mul_1Mulgradients_1/pi/pow_2_grad/mul3gradients_2/gradients_1/pi/pow_2_grad/Sum_grad/Tile*
T0*'
_output_shapes
:?????????
?
6gradients_2/gradients_1/pi/pow_2_grad/mul_1_grad/Sum_1Sum6gradients_2/gradients_1/pi/pow_2_grad/mul_1_grad/Mul_1Hgradients_2/gradients_1/pi/pow_2_grad/mul_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*

Tidx0*
	keep_dims( 
?
:gradients_2/gradients_1/pi/pow_2_grad/mul_1_grad/Reshape_1Reshape6gradients_2/gradients_1/pi/pow_2_grad/mul_1_grad/Sum_18gradients_2/gradients_1/pi/pow_2_grad/mul_1_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:?????????
|
4gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/ShapeShapepi/sub_2*
_output_shapes
:*
T0*
out_type0
y
6gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
?
Dgradients_2/gradients_1/pi/pow_2_grad/Pow_grad/BroadcastGradientArgsBroadcastGradientArgs4gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/Shape6gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
2gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/mulMul:gradients_2/gradients_1/pi/pow_2_grad/mul_1_grad/Reshape_1gradients_1/pi/pow_2_grad/sub*
T0*'
_output_shapes
:?????????
y
4gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/sub/yConst*
valueB
 *  ??*
_output_shapes
: *
dtype0
?
2gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/subSubgradients_1/pi/pow_2_grad/sub4gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/sub/y*
T0*
_output_shapes
: 
?
2gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/PowPowpi/sub_22gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/sub*
T0*'
_output_shapes
:?????????
?
4gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/mul_1Mul2gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/mul2gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/Pow*'
_output_shapes
:?????????*
T0
?
2gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/SumSum4gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/mul_1Dgradients_2/gradients_1/pi/pow_2_grad/Pow_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
?
6gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/ReshapeReshape2gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/Sum4gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/Shape*'
_output_shapes
:?????????*
Tshape0*
T0
}
8gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/Greater/yConst*
valueB
 *    *
_output_shapes
: *
dtype0
?
6gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/GreaterGreaterpi/sub_28gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/Greater/y*
T0*'
_output_shapes
:?????????
?
>gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/ones_like/ShapeShapepi/sub_2*
T0*
_output_shapes
:*
out_type0
?
>gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/ones_like/ConstConst*
_output_shapes
: *
valueB
 *  ??*
dtype0
?
8gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/ones_likeFill>gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/ones_like/Shape>gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/ones_like/Const*

index_type0*
T0*'
_output_shapes
:?????????
?
5gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/SelectSelect6gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/Greaterpi/sub_28gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/ones_like*
T0*'
_output_shapes
:?????????
?
2gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/LogLog5gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/Select*'
_output_shapes
:?????????*
T0
?
9gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/zeros_like	ZerosLikepi/sub_2*'
_output_shapes
:?????????*
T0
?
7gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/Select_1Select6gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/Greater2gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/Log9gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/zeros_like*'
_output_shapes
:?????????*
T0
?
4gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/mul_2Mul:gradients_2/gradients_1/pi/pow_2_grad/mul_1_grad/Reshape_1gradients_1/pi/pow_2_grad/Pow*'
_output_shapes
:?????????*
T0
?
4gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/mul_3Mul4gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/mul_27gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/Select_1*'
_output_shapes
:?????????*
T0
?
4gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/Sum_1Sum4gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/mul_3Fgradients_2/gradients_1/pi/pow_2_grad/Pow_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
T0*
	keep_dims( 
?
8gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/Reshape_1Reshape4gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/Sum_16gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/Shape_1*
Tshape0*
_output_shapes
: *
T0
m
gradients_2/pi/sub_2_grad/ShapeShapepi/Placeholder*
out_type0*
T0*
_output_shapes
:
s
!gradients_2/pi/sub_2_grad/Shape_1Shapepi/dense_2/BiasAdd*
T0*
_output_shapes
:*
out_type0
?
/gradients_2/pi/sub_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_2/pi/sub_2_grad/Shape!gradients_2/pi/sub_2_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
gradients_2/pi/sub_2_grad/SumSum6gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/Reshape/gradients_2/pi/sub_2_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
T0*
	keep_dims( 
?
!gradients_2/pi/sub_2_grad/ReshapeReshapegradients_2/pi/sub_2_grad/Sumgradients_2/pi/sub_2_grad/Shape*
T0*
Tshape0*'
_output_shapes
:?????????
?
gradients_2/pi/sub_2_grad/Sum_1Sum6gradients_2/gradients_1/pi/pow_2_grad/Pow_grad/Reshape1gradients_2/pi/sub_2_grad/BroadcastGradientArgs:1*
_output_shapes
:*
T0*

Tidx0*
	keep_dims( 
h
gradients_2/pi/sub_2_grad/NegNeggradients_2/pi/sub_2_grad/Sum_1*
T0*
_output_shapes
:
?
#gradients_2/pi/sub_2_grad/Reshape_1Reshapegradients_2/pi/sub_2_grad/Neg!gradients_2/pi/sub_2_grad/Shape_1*
Tshape0*'
_output_shapes
:?????????*
T0
?
/gradients_2/pi/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad#gradients_2/pi/sub_2_grad/Reshape_1*
data_formatNHWC*
T0*
_output_shapes
:
?
)gradients_2/pi/dense_2/MatMul_grad/MatMulMatMul#gradients_2/pi/sub_2_grad/Reshape_1pi/dense_2/kernel/read*
transpose_b(*
T0*
transpose_a( *(
_output_shapes
:??????????
?
+gradients_2/pi/dense_2/MatMul_grad/MatMul_1MatMulpi/dense_1/Tanh#gradients_2/pi/sub_2_grad/Reshape_1*
transpose_b( *
_output_shapes
:	?*
T0*
transpose_a(
?
gradients_2/AddN_3AddNCgradients_2/gradients_1/pi/dense_2/MatMul_grad/MatMul_1_grad/MatMul@gradients_2/gradients_1/pi/dense_1/Tanh_grad/TanhGrad_grad/mul_2)gradients_2/pi/dense_2/MatMul_grad/MatMul*
N*(
_output_shapes
:??????????*
T0*V
_classL
JHloc:@gradients_2/gradients_1/pi/dense_2/MatMul_grad/MatMul_1_grad/MatMul
?
)gradients_2/pi/dense_1/Tanh_grad/TanhGradTanhGradpi/dense_1/Tanhgradients_2/AddN_3*
T0*(
_output_shapes
:??????????
?
gradients_2/AddN_4AddNCgradients_2/gradients_1/pi/dense_2/MatMul_grad/MatMul_grad/MatMul_1+gradients_2/pi/dense_2/MatMul_grad/MatMul_1*
_output_shapes
:	?*V
_classL
JHloc:@gradients_2/gradients_1/pi/dense_2/MatMul_grad/MatMul_grad/MatMul_1*
T0*
N
?
/gradients_2/pi/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad)gradients_2/pi/dense_1/Tanh_grad/TanhGrad*
_output_shapes	
:?*
T0*
data_formatNHWC
?
)gradients_2/pi/dense_1/MatMul_grad/MatMulMatMul)gradients_2/pi/dense_1/Tanh_grad/TanhGradpi/dense_1/kernel/read*
transpose_b(*
transpose_a( *
T0*(
_output_shapes
:??????????
?
+gradients_2/pi/dense_1/MatMul_grad/MatMul_1MatMulpi/dense/Tanh)gradients_2/pi/dense_1/Tanh_grad/TanhGrad*
T0*
transpose_b( * 
_output_shapes
:
??*
transpose_a(
?
gradients_2/AddN_5AddNCgradients_2/gradients_1/pi/dense_1/MatMul_grad/MatMul_1_grad/MatMul>gradients_2/gradients_1/pi/dense/Tanh_grad/TanhGrad_grad/mul_2)gradients_2/pi/dense_1/MatMul_grad/MatMul*V
_classL
JHloc:@gradients_2/gradients_1/pi/dense_1/MatMul_grad/MatMul_1_grad/MatMul*
T0*(
_output_shapes
:??????????*
N
?
'gradients_2/pi/dense/Tanh_grad/TanhGradTanhGradpi/dense/Tanhgradients_2/AddN_5*(
_output_shapes
:??????????*
T0
?
gradients_2/AddN_6AddNCgradients_2/gradients_1/pi/dense_1/MatMul_grad/MatMul_grad/MatMul_1+gradients_2/pi/dense_1/MatMul_grad/MatMul_1*
N* 
_output_shapes
:
??*
T0*V
_classL
JHloc:@gradients_2/gradients_1/pi/dense_1/MatMul_grad/MatMul_grad/MatMul_1
?
-gradients_2/pi/dense/BiasAdd_grad/BiasAddGradBiasAddGrad'gradients_2/pi/dense/Tanh_grad/TanhGrad*
data_formatNHWC*
T0*
_output_shapes	
:?
?
'gradients_2/pi/dense/MatMul_grad/MatMulMatMul'gradients_2/pi/dense/Tanh_grad/TanhGradpi/dense/kernel/read*
T0*'
_output_shapes
:?????????<*
transpose_b(*
transpose_a( 
?
)gradients_2/pi/dense/MatMul_grad/MatMul_1MatMulPlaceholder'gradients_2/pi/dense/Tanh_grad/TanhGrad*
T0*
_output_shapes
:	<?*
transpose_a(*
transpose_b( 
c
Reshape_14/shapeConst*
valueB:
?????????*
dtype0*
_output_shapes
:
?

Reshape_14Reshape)gradients_2/pi/dense/MatMul_grad/MatMul_1Reshape_14/shape*
T0*
_output_shapes	
:?x*
Tshape0
c
Reshape_15/shapeConst*
dtype0*
_output_shapes
:*
valueB:
?????????
?

Reshape_15Reshape-gradients_2/pi/dense/BiasAdd_grad/BiasAddGradReshape_15/shape*
Tshape0*
_output_shapes	
:?*
T0
c
Reshape_16/shapeConst*
_output_shapes
:*
valueB:
?????????*
dtype0
p

Reshape_16Reshapegradients_2/AddN_6Reshape_16/shape*
Tshape0*
_output_shapes

:??*
T0
c
Reshape_17/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
?

Reshape_17Reshape/gradients_2/pi/dense_1/BiasAdd_grad/BiasAddGradReshape_17/shape*
T0*
Tshape0*
_output_shapes	
:?
c
Reshape_18/shapeConst*
valueB:
?????????*
dtype0*
_output_shapes
:
o

Reshape_18Reshapegradients_2/AddN_4Reshape_18/shape*
Tshape0*
_output_shapes	
:?*
T0
c
Reshape_19/shapeConst*
valueB:
?????????*
_output_shapes
:*
dtype0
?

Reshape_19Reshape/gradients_2/pi/dense_2/BiasAdd_grad/BiasAddGradReshape_19/shape*
Tshape0*
_output_shapes
:*
T0
c
Reshape_20/shapeConst*
dtype0*
_output_shapes
:*
valueB:
?????????


Reshape_20Reshape#gradients_2/pi/mul_5_grad/Reshape_1Reshape_20/shape*
T0*
Tshape0*
_output_shapes
:
O
concat_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
concat_2ConcatV2
Reshape_14
Reshape_15
Reshape_16
Reshape_17
Reshape_18
Reshape_19
Reshape_20concat_2/axis*
T0*
_output_shapes

:??*
N*

Tidx0
L
mul_4/xConst*
_output_shapes
: *
dtype0*
valueB
 *???=
K
mul_4Mulmul_4/xPlaceholder_9*
_output_shapes

:??*
T0
D
add_1Addconcat_2mul_4*
T0*
_output_shapes

:??
T
gradients_3/ShapeConst*
_output_shapes
: *
valueB *
dtype0
Z
gradients_3/grad_ys_0Const*
_output_shapes
: *
valueB
 *  ??*
dtype0
u
gradients_3/FillFillgradients_3/Shapegradients_3/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
o
%gradients_3/Mean_2_grad/Reshape/shapeConst*
valueB:*
_output_shapes
:*
dtype0
?
gradients_3/Mean_2_grad/ReshapeReshapegradients_3/Fill%gradients_3/Mean_2_grad/Reshape/shape*
Tshape0*
_output_shapes
:*
T0
b
gradients_3/Mean_2_grad/ShapeShapemul_1*
T0*
_output_shapes
:*
out_type0
?
gradients_3/Mean_2_grad/TileTilegradients_3/Mean_2_grad/Reshapegradients_3/Mean_2_grad/Shape*
T0*#
_output_shapes
:?????????*

Tmultiples0
d
gradients_3/Mean_2_grad/Shape_1Shapemul_1*
_output_shapes
:*
out_type0*
T0
b
gradients_3/Mean_2_grad/Shape_2Const*
_output_shapes
: *
valueB *
dtype0
g
gradients_3/Mean_2_grad/ConstConst*
dtype0*
valueB: *
_output_shapes
:
?
gradients_3/Mean_2_grad/ProdProdgradients_3/Mean_2_grad/Shape_1gradients_3/Mean_2_grad/Const*
T0*
	keep_dims( *
_output_shapes
: *

Tidx0
i
gradients_3/Mean_2_grad/Const_1Const*
valueB: *
_output_shapes
:*
dtype0
?
gradients_3/Mean_2_grad/Prod_1Prodgradients_3/Mean_2_grad/Shape_2gradients_3/Mean_2_grad/Const_1*

Tidx0*
T0*
_output_shapes
: *
	keep_dims( 
c
!gradients_3/Mean_2_grad/Maximum/yConst*
_output_shapes
: *
value	B :*
dtype0
?
gradients_3/Mean_2_grad/MaximumMaximumgradients_3/Mean_2_grad/Prod_1!gradients_3/Mean_2_grad/Maximum/y*
T0*
_output_shapes
: 
?
 gradients_3/Mean_2_grad/floordivFloorDivgradients_3/Mean_2_grad/Prodgradients_3/Mean_2_grad/Maximum*
_output_shapes
: *
T0
?
gradients_3/Mean_2_grad/CastCast gradients_3/Mean_2_grad/floordiv*

DstT0*

SrcT0*
Truncate( *
_output_shapes
: 
?
gradients_3/Mean_2_grad/truedivRealDivgradients_3/Mean_2_grad/Tilegradients_3/Mean_2_grad/Cast*
T0*#
_output_shapes
:?????????
_
gradients_3/mul_1_grad/ShapeShapeExp*
out_type0*
T0*
_output_shapes
:
k
gradients_3/mul_1_grad/Shape_1ShapePlaceholder_3*
_output_shapes
:*
T0*
out_type0
?
,gradients_3/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_3/mul_1_grad/Shapegradients_3/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????

gradients_3/mul_1_grad/MulMulgradients_3/Mean_2_grad/truedivPlaceholder_3*#
_output_shapes
:?????????*
T0
?
gradients_3/mul_1_grad/SumSumgradients_3/mul_1_grad/Mul,gradients_3/mul_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
?
gradients_3/mul_1_grad/ReshapeReshapegradients_3/mul_1_grad/Sumgradients_3/mul_1_grad/Shape*
T0*
Tshape0*#
_output_shapes
:?????????
w
gradients_3/mul_1_grad/Mul_1MulExpgradients_3/Mean_2_grad/truediv*
T0*#
_output_shapes
:?????????
?
gradients_3/mul_1_grad/Sum_1Sumgradients_3/mul_1_grad/Mul_1.gradients_3/mul_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
?
 gradients_3/mul_1_grad/Reshape_1Reshapegradients_3/mul_1_grad/Sum_1gradients_3/mul_1_grad/Shape_1*#
_output_shapes
:?????????*
Tshape0*
T0
r
gradients_3/Exp_grad/mulMulgradients_3/mul_1_grad/ReshapeExp*
T0*#
_output_shapes
:?????????
`
gradients_3/sub_grad/ShapeShapepi/Sum*
out_type0*
T0*
_output_shapes
:
i
gradients_3/sub_grad/Shape_1ShapePlaceholder_6*
out_type0*
_output_shapes
:*
T0
?
*gradients_3/sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_3/sub_grad/Shapegradients_3/sub_grad/Shape_1*2
_output_shapes 
:?????????:?????????*
T0
?
gradients_3/sub_grad/SumSumgradients_3/Exp_grad/mul*gradients_3/sub_grad/BroadcastGradientArgs*
_output_shapes
:*

Tidx0*
T0*
	keep_dims( 
?
gradients_3/sub_grad/ReshapeReshapegradients_3/sub_grad/Sumgradients_3/sub_grad/Shape*
Tshape0*#
_output_shapes
:?????????*
T0
?
gradients_3/sub_grad/Sum_1Sumgradients_3/Exp_grad/mul,gradients_3/sub_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
_output_shapes
:*
T0
^
gradients_3/sub_grad/NegNeggradients_3/sub_grad/Sum_1*
T0*
_output_shapes
:
?
gradients_3/sub_grad/Reshape_1Reshapegradients_3/sub_grad/Neggradients_3/sub_grad/Shape_1*
Tshape0*#
_output_shapes
:?????????*
T0
e
gradients_3/pi/Sum_grad/ShapeShapepi/mul_2*
T0*
out_type0*
_output_shapes
:
?
gradients_3/pi/Sum_grad/SizeConst*
_output_shapes
: *0
_class&
$"loc:@gradients_3/pi/Sum_grad/Shape*
dtype0*
value	B :
?
gradients_3/pi/Sum_grad/addAddpi/Sum/reduction_indicesgradients_3/pi/Sum_grad/Size*
T0*0
_class&
$"loc:@gradients_3/pi/Sum_grad/Shape*
_output_shapes
: 
?
gradients_3/pi/Sum_grad/modFloorModgradients_3/pi/Sum_grad/addgradients_3/pi/Sum_grad/Size*
T0*
_output_shapes
: *0
_class&
$"loc:@gradients_3/pi/Sum_grad/Shape
?
gradients_3/pi/Sum_grad/Shape_1Const*
_output_shapes
: *0
_class&
$"loc:@gradients_3/pi/Sum_grad/Shape*
dtype0*
valueB 
?
#gradients_3/pi/Sum_grad/range/startConst*
value	B : *
_output_shapes
: *
dtype0*0
_class&
$"loc:@gradients_3/pi/Sum_grad/Shape
?
#gradients_3/pi/Sum_grad/range/deltaConst*
_output_shapes
: *
dtype0*0
_class&
$"loc:@gradients_3/pi/Sum_grad/Shape*
value	B :
?
gradients_3/pi/Sum_grad/rangeRange#gradients_3/pi/Sum_grad/range/startgradients_3/pi/Sum_grad/Size#gradients_3/pi/Sum_grad/range/delta*0
_class&
$"loc:@gradients_3/pi/Sum_grad/Shape*
_output_shapes
:*

Tidx0
?
"gradients_3/pi/Sum_grad/Fill/valueConst*
value	B :*
dtype0*
_output_shapes
: *0
_class&
$"loc:@gradients_3/pi/Sum_grad/Shape
?
gradients_3/pi/Sum_grad/FillFillgradients_3/pi/Sum_grad/Shape_1"gradients_3/pi/Sum_grad/Fill/value*

index_type0*
T0*0
_class&
$"loc:@gradients_3/pi/Sum_grad/Shape*
_output_shapes
: 
?
%gradients_3/pi/Sum_grad/DynamicStitchDynamicStitchgradients_3/pi/Sum_grad/rangegradients_3/pi/Sum_grad/modgradients_3/pi/Sum_grad/Shapegradients_3/pi/Sum_grad/Fill*
_output_shapes
:*
N*
T0*0
_class&
$"loc:@gradients_3/pi/Sum_grad/Shape
?
!gradients_3/pi/Sum_grad/Maximum/yConst*
_output_shapes
: *0
_class&
$"loc:@gradients_3/pi/Sum_grad/Shape*
value	B :*
dtype0
?
gradients_3/pi/Sum_grad/MaximumMaximum%gradients_3/pi/Sum_grad/DynamicStitch!gradients_3/pi/Sum_grad/Maximum/y*
T0*0
_class&
$"loc:@gradients_3/pi/Sum_grad/Shape*
_output_shapes
:
?
 gradients_3/pi/Sum_grad/floordivFloorDivgradients_3/pi/Sum_grad/Shapegradients_3/pi/Sum_grad/Maximum*0
_class&
$"loc:@gradients_3/pi/Sum_grad/Shape*
_output_shapes
:*
T0
?
gradients_3/pi/Sum_grad/ReshapeReshapegradients_3/sub_grad/Reshape%gradients_3/pi/Sum_grad/DynamicStitch*
T0*
Tshape0*0
_output_shapes
:??????????????????
?
gradients_3/pi/Sum_grad/TileTilegradients_3/pi/Sum_grad/Reshape gradients_3/pi/Sum_grad/floordiv*
T0*'
_output_shapes
:?????????*

Tmultiples0
b
gradients_3/pi/mul_2_grad/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
i
!gradients_3/pi/mul_2_grad/Shape_1Shapepi/add_3*
out_type0*
T0*
_output_shapes
:
?
/gradients_3/pi/mul_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_3/pi/mul_2_grad/Shape!gradients_3/pi/mul_2_grad/Shape_1*2
_output_shapes 
:?????????:?????????*
T0
~
gradients_3/pi/mul_2_grad/MulMulgradients_3/pi/Sum_grad/Tilepi/add_3*
T0*'
_output_shapes
:?????????
?
gradients_3/pi/mul_2_grad/SumSumgradients_3/pi/mul_2_grad/Mul/gradients_3/pi/mul_2_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
_output_shapes
:*
T0
?
!gradients_3/pi/mul_2_grad/ReshapeReshapegradients_3/pi/mul_2_grad/Sumgradients_3/pi/mul_2_grad/Shape*
T0*
Tshape0*
_output_shapes
: 
?
gradients_3/pi/mul_2_grad/Mul_1Mul
pi/mul_2/xgradients_3/pi/Sum_grad/Tile*'
_output_shapes
:?????????*
T0
?
gradients_3/pi/mul_2_grad/Sum_1Sumgradients_3/pi/mul_2_grad/Mul_11gradients_3/pi/mul_2_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
?
#gradients_3/pi/mul_2_grad/Reshape_1Reshapegradients_3/pi/mul_2_grad/Sum_1!gradients_3/pi/mul_2_grad/Shape_1*'
_output_shapes
:?????????*
T0*
Tshape0
g
gradients_3/pi/add_3_grad/ShapeShapepi/add_2*
out_type0*
T0*
_output_shapes
:
d
!gradients_3/pi/add_3_grad/Shape_1Const*
valueB *
dtype0*
_output_shapes
: 
?
/gradients_3/pi/add_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_3/pi/add_3_grad/Shape!gradients_3/pi/add_3_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
gradients_3/pi/add_3_grad/SumSum#gradients_3/pi/mul_2_grad/Reshape_1/gradients_3/pi/add_3_grad/BroadcastGradientArgs*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
?
!gradients_3/pi/add_3_grad/ReshapeReshapegradients_3/pi/add_3_grad/Sumgradients_3/pi/add_3_grad/Shape*
T0*
Tshape0*'
_output_shapes
:?????????
?
gradients_3/pi/add_3_grad/Sum_1Sum#gradients_3/pi/mul_2_grad/Reshape_11gradients_3/pi/add_3_grad/BroadcastGradientArgs:1*

Tidx0*
_output_shapes
:*
	keep_dims( *
T0
?
#gradients_3/pi/add_3_grad/Reshape_1Reshapegradients_3/pi/add_3_grad/Sum_1!gradients_3/pi/add_3_grad/Shape_1*
_output_shapes
: *
T0*
Tshape0
e
gradients_3/pi/add_2_grad/ShapeShapepi/pow*
T0*
out_type0*
_output_shapes
:
k
!gradients_3/pi/add_2_grad/Shape_1Const*
dtype0*
valueB:*
_output_shapes
:
?
/gradients_3/pi/add_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_3/pi/add_2_grad/Shape!gradients_3/pi/add_2_grad/Shape_1*2
_output_shapes 
:?????????:?????????*
T0
?
gradients_3/pi/add_2_grad/SumSum!gradients_3/pi/add_3_grad/Reshape/gradients_3/pi/add_2_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
?
!gradients_3/pi/add_2_grad/ReshapeReshapegradients_3/pi/add_2_grad/Sumgradients_3/pi/add_2_grad/Shape*
Tshape0*'
_output_shapes
:?????????*
T0
?
gradients_3/pi/add_2_grad/Sum_1Sum!gradients_3/pi/add_3_grad/Reshape1gradients_3/pi/add_2_grad/BroadcastGradientArgs:1*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
?
#gradients_3/pi/add_2_grad/Reshape_1Reshapegradients_3/pi/add_2_grad/Sum_1!gradients_3/pi/add_2_grad/Shape_1*
_output_shapes
:*
T0*
Tshape0
g
gradients_3/pi/pow_grad/ShapeShape
pi/truediv*
T0*
out_type0*
_output_shapes
:
b
gradients_3/pi/pow_grad/Shape_1Const*
valueB *
_output_shapes
: *
dtype0
?
-gradients_3/pi/pow_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_3/pi/pow_grad/Shapegradients_3/pi/pow_grad/Shape_1*2
_output_shapes 
:?????????:?????????*
T0
?
gradients_3/pi/pow_grad/mulMul!gradients_3/pi/add_2_grad/Reshapepi/pow/y*
T0*'
_output_shapes
:?????????
b
gradients_3/pi/pow_grad/sub/yConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
l
gradients_3/pi/pow_grad/subSubpi/pow/ygradients_3/pi/pow_grad/sub/y*
T0*
_output_shapes
: 
}
gradients_3/pi/pow_grad/PowPow
pi/truedivgradients_3/pi/pow_grad/sub*'
_output_shapes
:?????????*
T0
?
gradients_3/pi/pow_grad/mul_1Mulgradients_3/pi/pow_grad/mulgradients_3/pi/pow_grad/Pow*
T0*'
_output_shapes
:?????????
?
gradients_3/pi/pow_grad/SumSumgradients_3/pi/pow_grad/mul_1-gradients_3/pi/pow_grad/BroadcastGradientArgs*
	keep_dims( *
_output_shapes
:*
T0*

Tidx0
?
gradients_3/pi/pow_grad/ReshapeReshapegradients_3/pi/pow_grad/Sumgradients_3/pi/pow_grad/Shape*
Tshape0*
T0*'
_output_shapes
:?????????
f
!gradients_3/pi/pow_grad/Greater/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 
?
gradients_3/pi/pow_grad/GreaterGreater
pi/truediv!gradients_3/pi/pow_grad/Greater/y*
T0*'
_output_shapes
:?????????
q
'gradients_3/pi/pow_grad/ones_like/ShapeShape
pi/truediv*
T0*
_output_shapes
:*
out_type0
l
'gradients_3/pi/pow_grad/ones_like/ConstConst*
dtype0*
valueB
 *  ??*
_output_shapes
: 
?
!gradients_3/pi/pow_grad/ones_likeFill'gradients_3/pi/pow_grad/ones_like/Shape'gradients_3/pi/pow_grad/ones_like/Const*
T0*

index_type0*'
_output_shapes
:?????????
?
gradients_3/pi/pow_grad/SelectSelectgradients_3/pi/pow_grad/Greater
pi/truediv!gradients_3/pi/pow_grad/ones_like*'
_output_shapes
:?????????*
T0
t
gradients_3/pi/pow_grad/LogLoggradients_3/pi/pow_grad/Select*'
_output_shapes
:?????????*
T0
m
"gradients_3/pi/pow_grad/zeros_like	ZerosLike
pi/truediv*'
_output_shapes
:?????????*
T0
?
 gradients_3/pi/pow_grad/Select_1Selectgradients_3/pi/pow_grad/Greatergradients_3/pi/pow_grad/Log"gradients_3/pi/pow_grad/zeros_like*'
_output_shapes
:?????????*
T0
?
gradients_3/pi/pow_grad/mul_2Mul!gradients_3/pi/add_2_grad/Reshapepi/pow*
T0*'
_output_shapes
:?????????
?
gradients_3/pi/pow_grad/mul_3Mulgradients_3/pi/pow_grad/mul_2 gradients_3/pi/pow_grad/Select_1*
T0*'
_output_shapes
:?????????
?
gradients_3/pi/pow_grad/Sum_1Sumgradients_3/pi/pow_grad/mul_3/gradients_3/pi/pow_grad/BroadcastGradientArgs:1*
_output_shapes
:*

Tidx0*
	keep_dims( *
T0
?
!gradients_3/pi/pow_grad/Reshape_1Reshapegradients_3/pi/pow_grad/Sum_1gradients_3/pi/pow_grad/Shape_1*
Tshape0*
T0*
_output_shapes
: 
b
gradients_3/pi/mul_1_grad/ShapeConst*
valueB *
_output_shapes
: *
dtype0
k
!gradients_3/pi/mul_1_grad/Shape_1Const*
dtype0*
_output_shapes
:*
valueB:
?
/gradients_3/pi/mul_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_3/pi/mul_1_grad/Shape!gradients_3/pi/mul_1_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????

gradients_3/pi/mul_1_grad/MulMul#gradients_3/pi/add_2_grad/Reshape_1pi/log_std/read*
T0*
_output_shapes
:
?
gradients_3/pi/mul_1_grad/SumSumgradients_3/pi/mul_1_grad/Mul/gradients_3/pi/mul_1_grad/BroadcastGradientArgs*
T0*

Tidx0*
	keep_dims( *
_output_shapes
: 
?
!gradients_3/pi/mul_1_grad/ReshapeReshapegradients_3/pi/mul_1_grad/Sumgradients_3/pi/mul_1_grad/Shape*
_output_shapes
: *
T0*
Tshape0
|
gradients_3/pi/mul_1_grad/Mul_1Mul
pi/mul_1/x#gradients_3/pi/add_2_grad/Reshape_1*
T0*
_output_shapes
:
?
gradients_3/pi/mul_1_grad/Sum_1Sumgradients_3/pi/mul_1_grad/Mul_11gradients_3/pi/mul_1_grad/BroadcastGradientArgs:1*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( 
?
#gradients_3/pi/mul_1_grad/Reshape_1Reshapegradients_3/pi/mul_1_grad/Sum_1!gradients_3/pi/mul_1_grad/Shape_1*
_output_shapes
:*
Tshape0*
T0
g
!gradients_3/pi/truediv_grad/ShapeShapepi/sub*
T0*
_output_shapes
:*
out_type0
m
#gradients_3/pi/truediv_grad/Shape_1Const*
valueB:*
_output_shapes
:*
dtype0
?
1gradients_3/pi/truediv_grad/BroadcastGradientArgsBroadcastGradientArgs!gradients_3/pi/truediv_grad/Shape#gradients_3/pi/truediv_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
#gradients_3/pi/truediv_grad/RealDivRealDivgradients_3/pi/pow_grad/Reshapepi/add_1*
T0*'
_output_shapes
:?????????
?
gradients_3/pi/truediv_grad/SumSum#gradients_3/pi/truediv_grad/RealDiv1gradients_3/pi/truediv_grad/BroadcastGradientArgs*
	keep_dims( *
_output_shapes
:*

Tidx0*
T0
?
#gradients_3/pi/truediv_grad/ReshapeReshapegradients_3/pi/truediv_grad/Sum!gradients_3/pi/truediv_grad/Shape*
T0*'
_output_shapes
:?????????*
Tshape0
`
gradients_3/pi/truediv_grad/NegNegpi/sub*'
_output_shapes
:?????????*
T0
?
%gradients_3/pi/truediv_grad/RealDiv_1RealDivgradients_3/pi/truediv_grad/Negpi/add_1*'
_output_shapes
:?????????*
T0
?
%gradients_3/pi/truediv_grad/RealDiv_2RealDiv%gradients_3/pi/truediv_grad/RealDiv_1pi/add_1*
T0*'
_output_shapes
:?????????
?
gradients_3/pi/truediv_grad/mulMulgradients_3/pi/pow_grad/Reshape%gradients_3/pi/truediv_grad/RealDiv_2*'
_output_shapes
:?????????*
T0
?
!gradients_3/pi/truediv_grad/Sum_1Sumgradients_3/pi/truediv_grad/mul3gradients_3/pi/truediv_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
_output_shapes
:*
	keep_dims( 
?
%gradients_3/pi/truediv_grad/Reshape_1Reshape!gradients_3/pi/truediv_grad/Sum_1#gradients_3/pi/truediv_grad/Shape_1*
Tshape0*
_output_shapes
:*
T0
j
gradients_3/pi/sub_grad/ShapeShapePlaceholder_1*
T0*
_output_shapes
:*
out_type0
q
gradients_3/pi/sub_grad/Shape_1Shapepi/dense_2/BiasAdd*
_output_shapes
:*
out_type0*
T0
?
-gradients_3/pi/sub_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_3/pi/sub_grad/Shapegradients_3/pi/sub_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
gradients_3/pi/sub_grad/SumSum#gradients_3/pi/truediv_grad/Reshape-gradients_3/pi/sub_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*

Tidx0*
	keep_dims( 
?
gradients_3/pi/sub_grad/ReshapeReshapegradients_3/pi/sub_grad/Sumgradients_3/pi/sub_grad/Shape*
T0*'
_output_shapes
:?????????*
Tshape0
?
gradients_3/pi/sub_grad/Sum_1Sum#gradients_3/pi/truediv_grad/Reshape/gradients_3/pi/sub_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
d
gradients_3/pi/sub_grad/NegNeggradients_3/pi/sub_grad/Sum_1*
_output_shapes
:*
T0
?
!gradients_3/pi/sub_grad/Reshape_1Reshapegradients_3/pi/sub_grad/Neggradients_3/pi/sub_grad/Shape_1*'
_output_shapes
:?????????*
Tshape0*
T0
i
gradients_3/pi/add_1_grad/ShapeConst*
dtype0*
_output_shapes
:*
valueB:
d
!gradients_3/pi/add_1_grad/Shape_1Const*
valueB *
_output_shapes
: *
dtype0
?
/gradients_3/pi/add_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_3/pi/add_1_grad/Shape!gradients_3/pi/add_1_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
?
gradients_3/pi/add_1_grad/SumSum%gradients_3/pi/truediv_grad/Reshape_1/gradients_3/pi/add_1_grad/BroadcastGradientArgs*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
?
!gradients_3/pi/add_1_grad/ReshapeReshapegradients_3/pi/add_1_grad/Sumgradients_3/pi/add_1_grad/Shape*
_output_shapes
:*
T0*
Tshape0
?
gradients_3/pi/add_1_grad/Sum_1Sum%gradients_3/pi/truediv_grad/Reshape_11gradients_3/pi/add_1_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
?
#gradients_3/pi/add_1_grad/Reshape_1Reshapegradients_3/pi/add_1_grad/Sum_1!gradients_3/pi/add_1_grad/Shape_1*
T0*
_output_shapes
: *
Tshape0
?
/gradients_3/pi/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad!gradients_3/pi/sub_grad/Reshape_1*
T0*
data_formatNHWC*
_output_shapes
:
v
gradients_3/pi/Exp_1_grad/mulMul!gradients_3/pi/add_1_grad/Reshapepi/Exp_1*
_output_shapes
:*
T0
?
)gradients_3/pi/dense_2/MatMul_grad/MatMulMatMul!gradients_3/pi/sub_grad/Reshape_1pi/dense_2/kernel/read*
transpose_a( *(
_output_shapes
:??????????*
transpose_b(*
T0
?
+gradients_3/pi/dense_2/MatMul_grad/MatMul_1MatMulpi/dense_1/Tanh!gradients_3/pi/sub_grad/Reshape_1*
transpose_b( *
T0*
_output_shapes
:	?*
transpose_a(
?
gradients_3/AddNAddN#gradients_3/pi/mul_1_grad/Reshape_1gradients_3/pi/Exp_1_grad/mul*
_output_shapes
:*
T0*
N*6
_class,
*(loc:@gradients_3/pi/mul_1_grad/Reshape_1
?
)gradients_3/pi/dense_1/Tanh_grad/TanhGradTanhGradpi/dense_1/Tanh)gradients_3/pi/dense_2/MatMul_grad/MatMul*
T0*(
_output_shapes
:??????????
?
/gradients_3/pi/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad)gradients_3/pi/dense_1/Tanh_grad/TanhGrad*
data_formatNHWC*
T0*
_output_shapes	
:?
?
)gradients_3/pi/dense_1/MatMul_grad/MatMulMatMul)gradients_3/pi/dense_1/Tanh_grad/TanhGradpi/dense_1/kernel/read*
transpose_b(*(
_output_shapes
:??????????*
transpose_a( *
T0
?
+gradients_3/pi/dense_1/MatMul_grad/MatMul_1MatMulpi/dense/Tanh)gradients_3/pi/dense_1/Tanh_grad/TanhGrad*
T0* 
_output_shapes
:
??*
transpose_a(*
transpose_b( 
?
'gradients_3/pi/dense/Tanh_grad/TanhGradTanhGradpi/dense/Tanh)gradients_3/pi/dense_1/MatMul_grad/MatMul*
T0*(
_output_shapes
:??????????
?
-gradients_3/pi/dense/BiasAdd_grad/BiasAddGradBiasAddGrad'gradients_3/pi/dense/Tanh_grad/TanhGrad*
T0*
_output_shapes	
:?*
data_formatNHWC
?
'gradients_3/pi/dense/MatMul_grad/MatMulMatMul'gradients_3/pi/dense/Tanh_grad/TanhGradpi/dense/kernel/read*
T0*'
_output_shapes
:?????????<*
transpose_a( *
transpose_b(
?
)gradients_3/pi/dense/MatMul_grad/MatMul_1MatMulPlaceholder'gradients_3/pi/dense/Tanh_grad/TanhGrad*
transpose_b( *
transpose_a(*
_output_shapes
:	<?*
T0
c
Reshape_21/shapeConst*
valueB:
?????????*
_output_shapes
:*
dtype0
?

Reshape_21Reshape)gradients_3/pi/dense/MatMul_grad/MatMul_1Reshape_21/shape*
_output_shapes	
:?x*
T0*
Tshape0
c
Reshape_22/shapeConst*
valueB:
?????????*
dtype0*
_output_shapes
:
?

Reshape_22Reshape-gradients_3/pi/dense/BiasAdd_grad/BiasAddGradReshape_22/shape*
_output_shapes	
:?*
T0*
Tshape0
c
Reshape_23/shapeConst*
dtype0*
valueB:
?????????*
_output_shapes
:
?

Reshape_23Reshape+gradients_3/pi/dense_1/MatMul_grad/MatMul_1Reshape_23/shape*
T0*
Tshape0*
_output_shapes

:??
c
Reshape_24/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
?

Reshape_24Reshape/gradients_3/pi/dense_1/BiasAdd_grad/BiasAddGradReshape_24/shape*
T0*
Tshape0*
_output_shapes	
:?
c
Reshape_25/shapeConst*
dtype0*
_output_shapes
:*
valueB:
?????????
?

Reshape_25Reshape+gradients_3/pi/dense_2/MatMul_grad/MatMul_1Reshape_25/shape*
T0*
_output_shapes	
:?*
Tshape0
c
Reshape_26/shapeConst*
dtype0*
valueB:
?????????*
_output_shapes
:
?

Reshape_26Reshape/gradients_3/pi/dense_2/BiasAdd_grad/BiasAddGradReshape_26/shape*
_output_shapes
:*
Tshape0*
T0
c
Reshape_27/shapeConst*
valueB:
?????????*
dtype0*
_output_shapes
:
l

Reshape_27Reshapegradients_3/AddNReshape_27/shape*
Tshape0*
T0*
_output_shapes
:
O
concat_3/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
concat_3ConcatV2
Reshape_21
Reshape_22
Reshape_23
Reshape_24
Reshape_25
Reshape_26
Reshape_27concat_3/axis*
N*
_output_shapes

:??*
T0*

Tidx0
c
Reshape_28/shapeConst*
dtype0*
valueB:
?????????*
_output_shapes
:
q

Reshape_28Reshapepi/dense/kernel/readReshape_28/shape*
T0*
Tshape0*
_output_shapes	
:?x
c
Reshape_29/shapeConst*
valueB:
?????????*
dtype0*
_output_shapes
:
o

Reshape_29Reshapepi/dense/bias/readReshape_29/shape*
Tshape0*
_output_shapes	
:?*
T0
c
Reshape_30/shapeConst*
_output_shapes
:*
valueB:
?????????*
dtype0
t

Reshape_30Reshapepi/dense_1/kernel/readReshape_30/shape*
T0*
Tshape0*
_output_shapes

:??
c
Reshape_31/shapeConst*
dtype0*
_output_shapes
:*
valueB:
?????????
q

Reshape_31Reshapepi/dense_1/bias/readReshape_31/shape*
T0*
_output_shapes	
:?*
Tshape0
c
Reshape_32/shapeConst*
_output_shapes
:*
valueB:
?????????*
dtype0
s

Reshape_32Reshapepi/dense_2/kernel/readReshape_32/shape*
_output_shapes	
:?*
Tshape0*
T0
c
Reshape_33/shapeConst*
dtype0*
_output_shapes
:*
valueB:
?????????
p

Reshape_33Reshapepi/dense_2/bias/readReshape_33/shape*
T0*
_output_shapes
:*
Tshape0
c
Reshape_34/shapeConst*
valueB:
?????????*
dtype0*
_output_shapes
:
k

Reshape_34Reshapepi/log_std/readReshape_34/shape*
_output_shapes
:*
T0*
Tshape0
O
concat_4/axisConst*
value	B : *
_output_shapes
: *
dtype0
?
concat_4ConcatV2
Reshape_28
Reshape_29
Reshape_30
Reshape_31
Reshape_32
Reshape_33
Reshape_34concat_4/axis*

Tidx0*
_output_shapes

:??*
N*
T0
l
Const_4Const*1
value(B&" <                    *
dtype0*
_output_shapes
:
Q
split/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
?
splitSplitVPlaceholder_9Const_4split/split_dim*
T0*

Tlen0*D
_output_shapes2
0:?x:?:??:?:?::*
	num_split
a
Reshape_35/shapeConst*
_output_shapes
:*
dtype0*
valueB"<      
f

Reshape_35ReshapesplitReshape_35/shape*
_output_shapes
:	<?*
T0*
Tshape0
[
Reshape_36/shapeConst*
valueB:?*
dtype0*
_output_shapes
:
d

Reshape_36Reshapesplit:1Reshape_36/shape*
T0*
Tshape0*
_output_shapes	
:?
a
Reshape_37/shapeConst*
dtype0*
_output_shapes
:*
valueB"      
i

Reshape_37Reshapesplit:2Reshape_37/shape* 
_output_shapes
:
??*
Tshape0*
T0
[
Reshape_38/shapeConst*
dtype0*
valueB:?*
_output_shapes
:
d

Reshape_38Reshapesplit:3Reshape_38/shape*
T0*
Tshape0*
_output_shapes	
:?
a
Reshape_39/shapeConst*
dtype0*
_output_shapes
:*
valueB"      
h

Reshape_39Reshapesplit:4Reshape_39/shape*
Tshape0*
T0*
_output_shapes
:	?
Z
Reshape_40/shapeConst*
_output_shapes
:*
dtype0*
valueB:
c

Reshape_40Reshapesplit:5Reshape_40/shape*
_output_shapes
:*
Tshape0*
T0
Z
Reshape_41/shapeConst*
valueB:*
dtype0*
_output_shapes
:
c

Reshape_41Reshapesplit:6Reshape_41/shape*
Tshape0*
T0*
_output_shapes
:
?
AssignAssignpi/dense/kernel
Reshape_35*"
_class
loc:@pi/dense/kernel*
validate_shape(*
_output_shapes
:	<?*
T0*
use_locking(
?
Assign_1Assignpi/dense/bias
Reshape_36*
validate_shape(*
_output_shapes	
:?*
T0* 
_class
loc:@pi/dense/bias*
use_locking(
?
Assign_2Assignpi/dense_1/kernel
Reshape_37*
T0*
validate_shape(*
use_locking(*$
_class
loc:@pi/dense_1/kernel* 
_output_shapes
:
??
?
Assign_3Assignpi/dense_1/bias
Reshape_38*
use_locking(*
T0*
_output_shapes	
:?*"
_class
loc:@pi/dense_1/bias*
validate_shape(
?
Assign_4Assignpi/dense_2/kernel
Reshape_39*
validate_shape(*
use_locking(*$
_class
loc:@pi/dense_2/kernel*
_output_shapes
:	?*
T0
?
Assign_5Assignpi/dense_2/bias
Reshape_40*
_output_shapes
:*
validate_shape(*"
_class
loc:@pi/dense_2/bias*
use_locking(*
T0
?
Assign_6Assign
pi/log_std
Reshape_41*
use_locking(*
_output_shapes
:*
validate_shape(*
T0*
_class
loc:@pi/log_std
]

group_depsNoOp^Assign	^Assign_1	^Assign_2	^Assign_3	^Assign_4	^Assign_5	^Assign_6
U
sub_1SubPlaceholder_4
vf/Squeeze*
T0*#
_output_shapes
:?????????
J
pow/yConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
F
powPowsub_1pow/y*
T0*#
_output_shapes
:?????????
Q
Const_5Const*
dtype0*
_output_shapes
:*
valueB: 
Z
Mean_3MeanpowConst_5*

Tidx0*
	keep_dims( *
T0*
_output_shapes
: 
U
sub_2SubPlaceholder_5
vc/Squeeze*
T0*#
_output_shapes
:?????????
L
pow_1/yConst*
valueB
 *   @*
_output_shapes
: *
dtype0
J
pow_1Powsub_2pow_1/y*#
_output_shapes
:?????????*
T0
Q
Const_6Const*
dtype0*
valueB: *
_output_shapes
:
\
Mean_4Meanpow_1Const_6*
T0*
_output_shapes
: *

Tidx0*
	keep_dims( 
=
add_2AddMean_3Mean_4*
T0*
_output_shapes
: 
T
gradients_4/ShapeConst*
valueB *
_output_shapes
: *
dtype0
Z
gradients_4/grad_ys_0Const*
dtype0*
_output_shapes
: *
valueB
 *  ??
u
gradients_4/FillFillgradients_4/Shapegradients_4/grad_ys_0*
_output_shapes
: *

index_type0*
T0
B
'gradients_4/add_2_grad/tuple/group_depsNoOp^gradients_4/Fill
?
/gradients_4/add_2_grad/tuple/control_dependencyIdentitygradients_4/Fill(^gradients_4/add_2_grad/tuple/group_deps*
T0*#
_class
loc:@gradients_4/Fill*
_output_shapes
: 
?
1gradients_4/add_2_grad/tuple/control_dependency_1Identitygradients_4/Fill(^gradients_4/add_2_grad/tuple/group_deps*#
_class
loc:@gradients_4/Fill*
_output_shapes
: *
T0
o
%gradients_4/Mean_3_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:
?
gradients_4/Mean_3_grad/ReshapeReshape/gradients_4/add_2_grad/tuple/control_dependency%gradients_4/Mean_3_grad/Reshape/shape*
Tshape0*
_output_shapes
:*
T0
`
gradients_4/Mean_3_grad/ShapeShapepow*
out_type0*
_output_shapes
:*
T0
?
gradients_4/Mean_3_grad/TileTilegradients_4/Mean_3_grad/Reshapegradients_4/Mean_3_grad/Shape*

Tmultiples0*#
_output_shapes
:?????????*
T0
b
gradients_4/Mean_3_grad/Shape_1Shapepow*
_output_shapes
:*
out_type0*
T0
b
gradients_4/Mean_3_grad/Shape_2Const*
_output_shapes
: *
valueB *
dtype0
g
gradients_4/Mean_3_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
?
gradients_4/Mean_3_grad/ProdProdgradients_4/Mean_3_grad/Shape_1gradients_4/Mean_3_grad/Const*

Tidx0*
_output_shapes
: *
	keep_dims( *
T0
i
gradients_4/Mean_3_grad/Const_1Const*
valueB: *
_output_shapes
:*
dtype0
?
gradients_4/Mean_3_grad/Prod_1Prodgradients_4/Mean_3_grad/Shape_2gradients_4/Mean_3_grad/Const_1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
: 
c
!gradients_4/Mean_3_grad/Maximum/yConst*
dtype0*
_output_shapes
: *
value	B :
?
gradients_4/Mean_3_grad/MaximumMaximumgradients_4/Mean_3_grad/Prod_1!gradients_4/Mean_3_grad/Maximum/y*
T0*
_output_shapes
: 
?
 gradients_4/Mean_3_grad/floordivFloorDivgradients_4/Mean_3_grad/Prodgradients_4/Mean_3_grad/Maximum*
T0*
_output_shapes
: 
?
gradients_4/Mean_3_grad/CastCast gradients_4/Mean_3_grad/floordiv*

DstT0*
Truncate( *
_output_shapes
: *

SrcT0
?
gradients_4/Mean_3_grad/truedivRealDivgradients_4/Mean_3_grad/Tilegradients_4/Mean_3_grad/Cast*#
_output_shapes
:?????????*
T0
o
%gradients_4/Mean_4_grad/Reshape/shapeConst*
valueB:*
_output_shapes
:*
dtype0
?
gradients_4/Mean_4_grad/ReshapeReshape1gradients_4/add_2_grad/tuple/control_dependency_1%gradients_4/Mean_4_grad/Reshape/shape*
_output_shapes
:*
Tshape0*
T0
b
gradients_4/Mean_4_grad/ShapeShapepow_1*
T0*
_output_shapes
:*
out_type0
?
gradients_4/Mean_4_grad/TileTilegradients_4/Mean_4_grad/Reshapegradients_4/Mean_4_grad/Shape*#
_output_shapes
:?????????*
T0*

Tmultiples0
d
gradients_4/Mean_4_grad/Shape_1Shapepow_1*
_output_shapes
:*
T0*
out_type0
b
gradients_4/Mean_4_grad/Shape_2Const*
_output_shapes
: *
valueB *
dtype0
g
gradients_4/Mean_4_grad/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
?
gradients_4/Mean_4_grad/ProdProdgradients_4/Mean_4_grad/Shape_1gradients_4/Mean_4_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
i
gradients_4/Mean_4_grad/Const_1Const*
_output_shapes
:*
valueB: *
dtype0
?
gradients_4/Mean_4_grad/Prod_1Prodgradients_4/Mean_4_grad/Shape_2gradients_4/Mean_4_grad/Const_1*
_output_shapes
: *
T0*
	keep_dims( *

Tidx0
c
!gradients_4/Mean_4_grad/Maximum/yConst*
_output_shapes
: *
value	B :*
dtype0
?
gradients_4/Mean_4_grad/MaximumMaximumgradients_4/Mean_4_grad/Prod_1!gradients_4/Mean_4_grad/Maximum/y*
_output_shapes
: *
T0
?
 gradients_4/Mean_4_grad/floordivFloorDivgradients_4/Mean_4_grad/Prodgradients_4/Mean_4_grad/Maximum*
_output_shapes
: *
T0
?
gradients_4/Mean_4_grad/CastCast gradients_4/Mean_4_grad/floordiv*

DstT0*
Truncate( *

SrcT0*
_output_shapes
: 
?
gradients_4/Mean_4_grad/truedivRealDivgradients_4/Mean_4_grad/Tilegradients_4/Mean_4_grad/Cast*#
_output_shapes
:?????????*
T0
_
gradients_4/pow_grad/ShapeShapesub_1*
out_type0*
T0*
_output_shapes
:
_
gradients_4/pow_grad/Shape_1Const*
dtype0*
valueB *
_output_shapes
: 
?
*gradients_4/pow_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_4/pow_grad/Shapegradients_4/pow_grad/Shape_1*2
_output_shapes 
:?????????:?????????*
T0
u
gradients_4/pow_grad/mulMulgradients_4/Mean_3_grad/truedivpow/y*
T0*#
_output_shapes
:?????????
_
gradients_4/pow_grad/sub/yConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
c
gradients_4/pow_grad/subSubpow/ygradients_4/pow_grad/sub/y*
_output_shapes
: *
T0
n
gradients_4/pow_grad/PowPowsub_1gradients_4/pow_grad/sub*
T0*#
_output_shapes
:?????????
?
gradients_4/pow_grad/mul_1Mulgradients_4/pow_grad/mulgradients_4/pow_grad/Pow*
T0*#
_output_shapes
:?????????
?
gradients_4/pow_grad/SumSumgradients_4/pow_grad/mul_1*gradients_4/pow_grad/BroadcastGradientArgs*
T0*
	keep_dims( *

Tidx0*
_output_shapes
:
?
gradients_4/pow_grad/ReshapeReshapegradients_4/pow_grad/Sumgradients_4/pow_grad/Shape*#
_output_shapes
:?????????*
T0*
Tshape0
c
gradients_4/pow_grad/Greater/yConst*
dtype0*
valueB
 *    *
_output_shapes
: 
|
gradients_4/pow_grad/GreaterGreatersub_1gradients_4/pow_grad/Greater/y*
T0*#
_output_shapes
:?????????
i
$gradients_4/pow_grad/ones_like/ShapeShapesub_1*
out_type0*
_output_shapes
:*
T0
i
$gradients_4/pow_grad/ones_like/ConstConst*
dtype0*
valueB
 *  ??*
_output_shapes
: 
?
gradients_4/pow_grad/ones_likeFill$gradients_4/pow_grad/ones_like/Shape$gradients_4/pow_grad/ones_like/Const*

index_type0*#
_output_shapes
:?????????*
T0
?
gradients_4/pow_grad/SelectSelectgradients_4/pow_grad/Greatersub_1gradients_4/pow_grad/ones_like*
T0*#
_output_shapes
:?????????
j
gradients_4/pow_grad/LogLoggradients_4/pow_grad/Select*#
_output_shapes
:?????????*
T0
a
gradients_4/pow_grad/zeros_like	ZerosLikesub_1*
T0*#
_output_shapes
:?????????
?
gradients_4/pow_grad/Select_1Selectgradients_4/pow_grad/Greatergradients_4/pow_grad/Loggradients_4/pow_grad/zeros_like*
T0*#
_output_shapes
:?????????
u
gradients_4/pow_grad/mul_2Mulgradients_4/Mean_3_grad/truedivpow*#
_output_shapes
:?????????*
T0
?
gradients_4/pow_grad/mul_3Mulgradients_4/pow_grad/mul_2gradients_4/pow_grad/Select_1*#
_output_shapes
:?????????*
T0
?
gradients_4/pow_grad/Sum_1Sumgradients_4/pow_grad/mul_3,gradients_4/pow_grad/BroadcastGradientArgs:1*
	keep_dims( *
_output_shapes
:*

Tidx0*
T0
?
gradients_4/pow_grad/Reshape_1Reshapegradients_4/pow_grad/Sum_1gradients_4/pow_grad/Shape_1*
Tshape0*
T0*
_output_shapes
: 
m
%gradients_4/pow_grad/tuple/group_depsNoOp^gradients_4/pow_grad/Reshape^gradients_4/pow_grad/Reshape_1
?
-gradients_4/pow_grad/tuple/control_dependencyIdentitygradients_4/pow_grad/Reshape&^gradients_4/pow_grad/tuple/group_deps*#
_output_shapes
:?????????*
T0*/
_class%
#!loc:@gradients_4/pow_grad/Reshape
?
/gradients_4/pow_grad/tuple/control_dependency_1Identitygradients_4/pow_grad/Reshape_1&^gradients_4/pow_grad/tuple/group_deps*1
_class'
%#loc:@gradients_4/pow_grad/Reshape_1*
T0*
_output_shapes
: 
a
gradients_4/pow_1_grad/ShapeShapesub_2*
T0*
_output_shapes
:*
out_type0
a
gradients_4/pow_1_grad/Shape_1Const*
_output_shapes
: *
dtype0*
valueB 
?
,gradients_4/pow_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_4/pow_1_grad/Shapegradients_4/pow_1_grad/Shape_1*
T0*2
_output_shapes 
:?????????:?????????
y
gradients_4/pow_1_grad/mulMulgradients_4/Mean_4_grad/truedivpow_1/y*#
_output_shapes
:?????????*
T0
a
gradients_4/pow_1_grad/sub/yConst*
valueB
 *  ??*
_output_shapes
: *
dtype0
i
gradients_4/pow_1_grad/subSubpow_1/ygradients_4/pow_1_grad/sub/y*
T0*
_output_shapes
: 
r
gradients_4/pow_1_grad/PowPowsub_2gradients_4/pow_1_grad/sub*#
_output_shapes
:?????????*
T0
?
gradients_4/pow_1_grad/mul_1Mulgradients_4/pow_1_grad/mulgradients_4/pow_1_grad/Pow*
T0*#
_output_shapes
:?????????
?
gradients_4/pow_1_grad/SumSumgradients_4/pow_1_grad/mul_1,gradients_4/pow_1_grad/BroadcastGradientArgs*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( 
?
gradients_4/pow_1_grad/ReshapeReshapegradients_4/pow_1_grad/Sumgradients_4/pow_1_grad/Shape*#
_output_shapes
:?????????*
Tshape0*
T0
e
 gradients_4/pow_1_grad/Greater/yConst*
valueB
 *    *
dtype0*
_output_shapes
: 
?
gradients_4/pow_1_grad/GreaterGreatersub_2 gradients_4/pow_1_grad/Greater/y*
T0*#
_output_shapes
:?????????
k
&gradients_4/pow_1_grad/ones_like/ShapeShapesub_2*
out_type0*
T0*
_output_shapes
:
k
&gradients_4/pow_1_grad/ones_like/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *  ??
?
 gradients_4/pow_1_grad/ones_likeFill&gradients_4/pow_1_grad/ones_like/Shape&gradients_4/pow_1_grad/ones_like/Const*
T0*#
_output_shapes
:?????????*

index_type0
?
gradients_4/pow_1_grad/SelectSelectgradients_4/pow_1_grad/Greatersub_2 gradients_4/pow_1_grad/ones_like*
T0*#
_output_shapes
:?????????
n
gradients_4/pow_1_grad/LogLoggradients_4/pow_1_grad/Select*#
_output_shapes
:?????????*
T0
c
!gradients_4/pow_1_grad/zeros_like	ZerosLikesub_2*#
_output_shapes
:?????????*
T0
?
gradients_4/pow_1_grad/Select_1Selectgradients_4/pow_1_grad/Greatergradients_4/pow_1_grad/Log!gradients_4/pow_1_grad/zeros_like*#
_output_shapes
:?????????*
T0
y
gradients_4/pow_1_grad/mul_2Mulgradients_4/Mean_4_grad/truedivpow_1*
T0*#
_output_shapes
:?????????
?
gradients_4/pow_1_grad/mul_3Mulgradients_4/pow_1_grad/mul_2gradients_4/pow_1_grad/Select_1*
T0*#
_output_shapes
:?????????
?
gradients_4/pow_1_grad/Sum_1Sumgradients_4/pow_1_grad/mul_3.gradients_4/pow_1_grad/BroadcastGradientArgs:1*
T0*

Tidx0*
	keep_dims( *
_output_shapes
:
?
 gradients_4/pow_1_grad/Reshape_1Reshapegradients_4/pow_1_grad/Sum_1gradients_4/pow_1_grad/Shape_1*
Tshape0*
T0*
_output_shapes
: 
s
'gradients_4/pow_1_grad/tuple/group_depsNoOp^gradients_4/pow_1_grad/Reshape!^gradients_4/pow_1_grad/Reshape_1
?
/gradients_4/pow_1_grad/tuple/control_dependencyIdentitygradients_4/pow_1_grad/Reshape(^gradients_4/pow_1_grad/tuple/group_deps*#
_output_shapes
:?????????*1
_class'
%#loc:@gradients_4/pow_1_grad/Reshape*
T0
?
1gradients_4/pow_1_grad/tuple/control_dependency_1Identity gradients_4/pow_1_grad/Reshape_1(^gradients_4/pow_1_grad/tuple/group_deps*
T0*
_output_shapes
: *3
_class)
'%loc:@gradients_4/pow_1_grad/Reshape_1
i
gradients_4/sub_1_grad/ShapeShapePlaceholder_4*
_output_shapes
:*
T0*
out_type0
h
gradients_4/sub_1_grad/Shape_1Shape
vf/Squeeze*
T0*
out_type0*
_output_shapes
:
?
,gradients_4/sub_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_4/sub_1_grad/Shapegradients_4/sub_1_grad/Shape_1*2
_output_shapes 
:?????????:?????????*
T0
?
gradients_4/sub_1_grad/SumSum-gradients_4/pow_grad/tuple/control_dependency,gradients_4/sub_1_grad/BroadcastGradientArgs*

Tidx0*
T0*
	keep_dims( *
_output_shapes
:
?
gradients_4/sub_1_grad/ReshapeReshapegradients_4/sub_1_grad/Sumgradients_4/sub_1_grad/Shape*
T0*#
_output_shapes
:?????????*
Tshape0
?
gradients_4/sub_1_grad/Sum_1Sum-gradients_4/pow_grad/tuple/control_dependency.gradients_4/sub_1_grad/BroadcastGradientArgs:1*

Tidx0*
_output_shapes
:*
T0*
	keep_dims( 
b
gradients_4/sub_1_grad/NegNeggradients_4/sub_1_grad/Sum_1*
_output_shapes
:*
T0
?
 gradients_4/sub_1_grad/Reshape_1Reshapegradients_4/sub_1_grad/Neggradients_4/sub_1_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:?????????
s
'gradients_4/sub_1_grad/tuple/group_depsNoOp^gradients_4/sub_1_grad/Reshape!^gradients_4/sub_1_grad/Reshape_1
?
/gradients_4/sub_1_grad/tuple/control_dependencyIdentitygradients_4/sub_1_grad/Reshape(^gradients_4/sub_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients_4/sub_1_grad/Reshape*#
_output_shapes
:?????????
?
1gradients_4/sub_1_grad/tuple/control_dependency_1Identity gradients_4/sub_1_grad/Reshape_1(^gradients_4/sub_1_grad/tuple/group_deps*#
_output_shapes
:?????????*3
_class)
'%loc:@gradients_4/sub_1_grad/Reshape_1*
T0
i
gradients_4/sub_2_grad/ShapeShapePlaceholder_5*
_output_shapes
:*
out_type0*
T0
h
gradients_4/sub_2_grad/Shape_1Shape
vc/Squeeze*
out_type0*
T0*
_output_shapes
:
?
,gradients_4/sub_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_4/sub_2_grad/Shapegradients_4/sub_2_grad/Shape_1*2
_output_shapes 
:?????????:?????????*
T0
?
gradients_4/sub_2_grad/SumSum/gradients_4/pow_1_grad/tuple/control_dependency,gradients_4/sub_2_grad/BroadcastGradientArgs*
	keep_dims( *
T0*
_output_shapes
:*

Tidx0
?
gradients_4/sub_2_grad/ReshapeReshapegradients_4/sub_2_grad/Sumgradients_4/sub_2_grad/Shape*
T0*#
_output_shapes
:?????????*
Tshape0
?
gradients_4/sub_2_grad/Sum_1Sum/gradients_4/pow_1_grad/tuple/control_dependency.gradients_4/sub_2_grad/BroadcastGradientArgs:1*

Tidx0*
	keep_dims( *
_output_shapes
:*
T0
b
gradients_4/sub_2_grad/NegNeggradients_4/sub_2_grad/Sum_1*
_output_shapes
:*
T0
?
 gradients_4/sub_2_grad/Reshape_1Reshapegradients_4/sub_2_grad/Neggradients_4/sub_2_grad/Shape_1*#
_output_shapes
:?????????*
Tshape0*
T0
s
'gradients_4/sub_2_grad/tuple/group_depsNoOp^gradients_4/sub_2_grad/Reshape!^gradients_4/sub_2_grad/Reshape_1
?
/gradients_4/sub_2_grad/tuple/control_dependencyIdentitygradients_4/sub_2_grad/Reshape(^gradients_4/sub_2_grad/tuple/group_deps*#
_output_shapes
:?????????*1
_class'
%#loc:@gradients_4/sub_2_grad/Reshape*
T0
?
1gradients_4/sub_2_grad/tuple/control_dependency_1Identity gradients_4/sub_2_grad/Reshape_1(^gradients_4/sub_2_grad/tuple/group_deps*3
_class)
'%loc:@gradients_4/sub_2_grad/Reshape_1*#
_output_shapes
:?????????*
T0
s
!gradients_4/vf/Squeeze_grad/ShapeShapevf/dense_2/BiasAdd*
out_type0*
_output_shapes
:*
T0
?
#gradients_4/vf/Squeeze_grad/ReshapeReshape1gradients_4/sub_1_grad/tuple/control_dependency_1!gradients_4/vf/Squeeze_grad/Shape*
T0*
Tshape0*'
_output_shapes
:?????????
s
!gradients_4/vc/Squeeze_grad/ShapeShapevc/dense_2/BiasAdd*
_output_shapes
:*
out_type0*
T0
?
#gradients_4/vc/Squeeze_grad/ReshapeReshape1gradients_4/sub_2_grad/tuple/control_dependency_1!gradients_4/vc/Squeeze_grad/Shape*
Tshape0*
T0*'
_output_shapes
:?????????
?
/gradients_4/vf/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad#gradients_4/vf/Squeeze_grad/Reshape*
data_formatNHWC*
T0*
_output_shapes
:
?
4gradients_4/vf/dense_2/BiasAdd_grad/tuple/group_depsNoOp$^gradients_4/vf/Squeeze_grad/Reshape0^gradients_4/vf/dense_2/BiasAdd_grad/BiasAddGrad
?
<gradients_4/vf/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity#gradients_4/vf/Squeeze_grad/Reshape5^gradients_4/vf/dense_2/BiasAdd_grad/tuple/group_deps*6
_class,
*(loc:@gradients_4/vf/Squeeze_grad/Reshape*
T0*'
_output_shapes
:?????????
?
>gradients_4/vf/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity/gradients_4/vf/dense_2/BiasAdd_grad/BiasAddGrad5^gradients_4/vf/dense_2/BiasAdd_grad/tuple/group_deps*B
_class8
64loc:@gradients_4/vf/dense_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:*
T0
?
/gradients_4/vc/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad#gradients_4/vc/Squeeze_grad/Reshape*
data_formatNHWC*
_output_shapes
:*
T0
?
4gradients_4/vc/dense_2/BiasAdd_grad/tuple/group_depsNoOp$^gradients_4/vc/Squeeze_grad/Reshape0^gradients_4/vc/dense_2/BiasAdd_grad/BiasAddGrad
?
<gradients_4/vc/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity#gradients_4/vc/Squeeze_grad/Reshape5^gradients_4/vc/dense_2/BiasAdd_grad/tuple/group_deps*'
_output_shapes
:?????????*
T0*6
_class,
*(loc:@gradients_4/vc/Squeeze_grad/Reshape
?
>gradients_4/vc/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity/gradients_4/vc/dense_2/BiasAdd_grad/BiasAddGrad5^gradients_4/vc/dense_2/BiasAdd_grad/tuple/group_deps*B
_class8
64loc:@gradients_4/vc/dense_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:*
T0
?
)gradients_4/vf/dense_2/MatMul_grad/MatMulMatMul<gradients_4/vf/dense_2/BiasAdd_grad/tuple/control_dependencyvf/dense_2/kernel/read*
T0*
transpose_a( *(
_output_shapes
:??????????*
transpose_b(
?
+gradients_4/vf/dense_2/MatMul_grad/MatMul_1MatMulvf/dense_1/Tanh<gradients_4/vf/dense_2/BiasAdd_grad/tuple/control_dependency*
_output_shapes
:	?*
transpose_b( *
T0*
transpose_a(
?
3gradients_4/vf/dense_2/MatMul_grad/tuple/group_depsNoOp*^gradients_4/vf/dense_2/MatMul_grad/MatMul,^gradients_4/vf/dense_2/MatMul_grad/MatMul_1
?
;gradients_4/vf/dense_2/MatMul_grad/tuple/control_dependencyIdentity)gradients_4/vf/dense_2/MatMul_grad/MatMul4^gradients_4/vf/dense_2/MatMul_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients_4/vf/dense_2/MatMul_grad/MatMul*(
_output_shapes
:??????????
?
=gradients_4/vf/dense_2/MatMul_grad/tuple/control_dependency_1Identity+gradients_4/vf/dense_2/MatMul_grad/MatMul_14^gradients_4/vf/dense_2/MatMul_grad/tuple/group_deps*>
_class4
20loc:@gradients_4/vf/dense_2/MatMul_grad/MatMul_1*
_output_shapes
:	?*
T0
?
)gradients_4/vc/dense_2/MatMul_grad/MatMulMatMul<gradients_4/vc/dense_2/BiasAdd_grad/tuple/control_dependencyvc/dense_2/kernel/read*
T0*(
_output_shapes
:??????????*
transpose_b(*
transpose_a( 
?
+gradients_4/vc/dense_2/MatMul_grad/MatMul_1MatMulvc/dense_1/Tanh<gradients_4/vc/dense_2/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
transpose_a(*
_output_shapes
:	?*
T0
?
3gradients_4/vc/dense_2/MatMul_grad/tuple/group_depsNoOp*^gradients_4/vc/dense_2/MatMul_grad/MatMul,^gradients_4/vc/dense_2/MatMul_grad/MatMul_1
?
;gradients_4/vc/dense_2/MatMul_grad/tuple/control_dependencyIdentity)gradients_4/vc/dense_2/MatMul_grad/MatMul4^gradients_4/vc/dense_2/MatMul_grad/tuple/group_deps*(
_output_shapes
:??????????*<
_class2
0.loc:@gradients_4/vc/dense_2/MatMul_grad/MatMul*
T0
?
=gradients_4/vc/dense_2/MatMul_grad/tuple/control_dependency_1Identity+gradients_4/vc/dense_2/MatMul_grad/MatMul_14^gradients_4/vc/dense_2/MatMul_grad/tuple/group_deps*
_output_shapes
:	?*>
_class4
20loc:@gradients_4/vc/dense_2/MatMul_grad/MatMul_1*
T0
?
)gradients_4/vf/dense_1/Tanh_grad/TanhGradTanhGradvf/dense_1/Tanh;gradients_4/vf/dense_2/MatMul_grad/tuple/control_dependency*(
_output_shapes
:??????????*
T0
?
)gradients_4/vc/dense_1/Tanh_grad/TanhGradTanhGradvc/dense_1/Tanh;gradients_4/vc/dense_2/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:??????????
?
/gradients_4/vf/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad)gradients_4/vf/dense_1/Tanh_grad/TanhGrad*
T0*
data_formatNHWC*
_output_shapes	
:?
?
4gradients_4/vf/dense_1/BiasAdd_grad/tuple/group_depsNoOp0^gradients_4/vf/dense_1/BiasAdd_grad/BiasAddGrad*^gradients_4/vf/dense_1/Tanh_grad/TanhGrad
?
<gradients_4/vf/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity)gradients_4/vf/dense_1/Tanh_grad/TanhGrad5^gradients_4/vf/dense_1/BiasAdd_grad/tuple/group_deps*<
_class2
0.loc:@gradients_4/vf/dense_1/Tanh_grad/TanhGrad*
T0*(
_output_shapes
:??????????
?
>gradients_4/vf/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity/gradients_4/vf/dense_1/BiasAdd_grad/BiasAddGrad5^gradients_4/vf/dense_1/BiasAdd_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients_4/vf/dense_1/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:?
?
/gradients_4/vc/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad)gradients_4/vc/dense_1/Tanh_grad/TanhGrad*
_output_shapes	
:?*
T0*
data_formatNHWC
?
4gradients_4/vc/dense_1/BiasAdd_grad/tuple/group_depsNoOp0^gradients_4/vc/dense_1/BiasAdd_grad/BiasAddGrad*^gradients_4/vc/dense_1/Tanh_grad/TanhGrad
?
<gradients_4/vc/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity)gradients_4/vc/dense_1/Tanh_grad/TanhGrad5^gradients_4/vc/dense_1/BiasAdd_grad/tuple/group_deps*<
_class2
0.loc:@gradients_4/vc/dense_1/Tanh_grad/TanhGrad*(
_output_shapes
:??????????*
T0
?
>gradients_4/vc/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity/gradients_4/vc/dense_1/BiasAdd_grad/BiasAddGrad5^gradients_4/vc/dense_1/BiasAdd_grad/tuple/group_deps*
T0*B
_class8
64loc:@gradients_4/vc/dense_1/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:?
?
)gradients_4/vf/dense_1/MatMul_grad/MatMulMatMul<gradients_4/vf/dense_1/BiasAdd_grad/tuple/control_dependencyvf/dense_1/kernel/read*
transpose_a( *(
_output_shapes
:??????????*
transpose_b(*
T0
?
+gradients_4/vf/dense_1/MatMul_grad/MatMul_1MatMulvf/dense/Tanh<gradients_4/vf/dense_1/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0* 
_output_shapes
:
??*
transpose_a(
?
3gradients_4/vf/dense_1/MatMul_grad/tuple/group_depsNoOp*^gradients_4/vf/dense_1/MatMul_grad/MatMul,^gradients_4/vf/dense_1/MatMul_grad/MatMul_1
?
;gradients_4/vf/dense_1/MatMul_grad/tuple/control_dependencyIdentity)gradients_4/vf/dense_1/MatMul_grad/MatMul4^gradients_4/vf/dense_1/MatMul_grad/tuple/group_deps*(
_output_shapes
:??????????*<
_class2
0.loc:@gradients_4/vf/dense_1/MatMul_grad/MatMul*
T0
?
=gradients_4/vf/dense_1/MatMul_grad/tuple/control_dependency_1Identity+gradients_4/vf/dense_1/MatMul_grad/MatMul_14^gradients_4/vf/dense_1/MatMul_grad/tuple/group_deps*>
_class4
20loc:@gradients_4/vf/dense_1/MatMul_grad/MatMul_1*
T0* 
_output_shapes
:
??
?
)gradients_4/vc/dense_1/MatMul_grad/MatMulMatMul<gradients_4/vc/dense_1/BiasAdd_grad/tuple/control_dependencyvc/dense_1/kernel/read*
T0*(
_output_shapes
:??????????*
transpose_b(*
transpose_a( 
?
+gradients_4/vc/dense_1/MatMul_grad/MatMul_1MatMulvc/dense/Tanh<gradients_4/vc/dense_1/BiasAdd_grad/tuple/control_dependency*
transpose_a(*
T0*
transpose_b( * 
_output_shapes
:
??
?
3gradients_4/vc/dense_1/MatMul_grad/tuple/group_depsNoOp*^gradients_4/vc/dense_1/MatMul_grad/MatMul,^gradients_4/vc/dense_1/MatMul_grad/MatMul_1
?
;gradients_4/vc/dense_1/MatMul_grad/tuple/control_dependencyIdentity)gradients_4/vc/dense_1/MatMul_grad/MatMul4^gradients_4/vc/dense_1/MatMul_grad/tuple/group_deps*
T0*<
_class2
0.loc:@gradients_4/vc/dense_1/MatMul_grad/MatMul*(
_output_shapes
:??????????
?
=gradients_4/vc/dense_1/MatMul_grad/tuple/control_dependency_1Identity+gradients_4/vc/dense_1/MatMul_grad/MatMul_14^gradients_4/vc/dense_1/MatMul_grad/tuple/group_deps* 
_output_shapes
:
??*>
_class4
20loc:@gradients_4/vc/dense_1/MatMul_grad/MatMul_1*
T0
?
'gradients_4/vf/dense/Tanh_grad/TanhGradTanhGradvf/dense/Tanh;gradients_4/vf/dense_1/MatMul_grad/tuple/control_dependency*
T0*(
_output_shapes
:??????????
?
'gradients_4/vc/dense/Tanh_grad/TanhGradTanhGradvc/dense/Tanh;gradients_4/vc/dense_1/MatMul_grad/tuple/control_dependency*(
_output_shapes
:??????????*
T0
?
-gradients_4/vf/dense/BiasAdd_grad/BiasAddGradBiasAddGrad'gradients_4/vf/dense/Tanh_grad/TanhGrad*
T0*
_output_shapes	
:?*
data_formatNHWC
?
2gradients_4/vf/dense/BiasAdd_grad/tuple/group_depsNoOp.^gradients_4/vf/dense/BiasAdd_grad/BiasAddGrad(^gradients_4/vf/dense/Tanh_grad/TanhGrad
?
:gradients_4/vf/dense/BiasAdd_grad/tuple/control_dependencyIdentity'gradients_4/vf/dense/Tanh_grad/TanhGrad3^gradients_4/vf/dense/BiasAdd_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients_4/vf/dense/Tanh_grad/TanhGrad*(
_output_shapes
:??????????
?
<gradients_4/vf/dense/BiasAdd_grad/tuple/control_dependency_1Identity-gradients_4/vf/dense/BiasAdd_grad/BiasAddGrad3^gradients_4/vf/dense/BiasAdd_grad/tuple/group_deps*@
_class6
42loc:@gradients_4/vf/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes	
:?*
T0
?
-gradients_4/vc/dense/BiasAdd_grad/BiasAddGradBiasAddGrad'gradients_4/vc/dense/Tanh_grad/TanhGrad*
data_formatNHWC*
_output_shapes	
:?*
T0
?
2gradients_4/vc/dense/BiasAdd_grad/tuple/group_depsNoOp.^gradients_4/vc/dense/BiasAdd_grad/BiasAddGrad(^gradients_4/vc/dense/Tanh_grad/TanhGrad
?
:gradients_4/vc/dense/BiasAdd_grad/tuple/control_dependencyIdentity'gradients_4/vc/dense/Tanh_grad/TanhGrad3^gradients_4/vc/dense/BiasAdd_grad/tuple/group_deps*(
_output_shapes
:??????????*:
_class0
.,loc:@gradients_4/vc/dense/Tanh_grad/TanhGrad*
T0
?
<gradients_4/vc/dense/BiasAdd_grad/tuple/control_dependency_1Identity-gradients_4/vc/dense/BiasAdd_grad/BiasAddGrad3^gradients_4/vc/dense/BiasAdd_grad/tuple/group_deps*
_output_shapes	
:?*
T0*@
_class6
42loc:@gradients_4/vc/dense/BiasAdd_grad/BiasAddGrad
?
'gradients_4/vf/dense/MatMul_grad/MatMulMatMul:gradients_4/vf/dense/BiasAdd_grad/tuple/control_dependencyvf/dense/kernel/read*
transpose_a( *
transpose_b(*
T0*'
_output_shapes
:?????????<
?
)gradients_4/vf/dense/MatMul_grad/MatMul_1MatMulPlaceholder:gradients_4/vf/dense/BiasAdd_grad/tuple/control_dependency*
transpose_a(*
T0*
_output_shapes
:	<?*
transpose_b( 
?
1gradients_4/vf/dense/MatMul_grad/tuple/group_depsNoOp(^gradients_4/vf/dense/MatMul_grad/MatMul*^gradients_4/vf/dense/MatMul_grad/MatMul_1
?
9gradients_4/vf/dense/MatMul_grad/tuple/control_dependencyIdentity'gradients_4/vf/dense/MatMul_grad/MatMul2^gradients_4/vf/dense/MatMul_grad/tuple/group_deps*'
_output_shapes
:?????????<*
T0*:
_class0
.,loc:@gradients_4/vf/dense/MatMul_grad/MatMul
?
;gradients_4/vf/dense/MatMul_grad/tuple/control_dependency_1Identity)gradients_4/vf/dense/MatMul_grad/MatMul_12^gradients_4/vf/dense/MatMul_grad/tuple/group_deps*
_output_shapes
:	<?*
T0*<
_class2
0.loc:@gradients_4/vf/dense/MatMul_grad/MatMul_1
?
'gradients_4/vc/dense/MatMul_grad/MatMulMatMul:gradients_4/vc/dense/BiasAdd_grad/tuple/control_dependencyvc/dense/kernel/read*'
_output_shapes
:?????????<*
transpose_b(*
T0*
transpose_a( 
?
)gradients_4/vc/dense/MatMul_grad/MatMul_1MatMulPlaceholder:gradients_4/vc/dense/BiasAdd_grad/tuple/control_dependency*
_output_shapes
:	<?*
T0*
transpose_b( *
transpose_a(
?
1gradients_4/vc/dense/MatMul_grad/tuple/group_depsNoOp(^gradients_4/vc/dense/MatMul_grad/MatMul*^gradients_4/vc/dense/MatMul_grad/MatMul_1
?
9gradients_4/vc/dense/MatMul_grad/tuple/control_dependencyIdentity'gradients_4/vc/dense/MatMul_grad/MatMul2^gradients_4/vc/dense/MatMul_grad/tuple/group_deps*
T0*:
_class0
.,loc:@gradients_4/vc/dense/MatMul_grad/MatMul*'
_output_shapes
:?????????<
?
;gradients_4/vc/dense/MatMul_grad/tuple/control_dependency_1Identity)gradients_4/vc/dense/MatMul_grad/MatMul_12^gradients_4/vc/dense/MatMul_grad/tuple/group_deps*
_output_shapes
:	<?*
T0*<
_class2
0.loc:@gradients_4/vc/dense/MatMul_grad/MatMul_1
c
Reshape_42/shapeConst*
dtype0*
_output_shapes
:*
valueB:
?????????
?

Reshape_42Reshape;gradients_4/vf/dense/MatMul_grad/tuple/control_dependency_1Reshape_42/shape*
T0*
Tshape0*
_output_shapes	
:?x
c
Reshape_43/shapeConst*
dtype0*
_output_shapes
:*
valueB:
?????????
?

Reshape_43Reshape<gradients_4/vf/dense/BiasAdd_grad/tuple/control_dependency_1Reshape_43/shape*
Tshape0*
T0*
_output_shapes	
:?
c
Reshape_44/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
?

Reshape_44Reshape=gradients_4/vf/dense_1/MatMul_grad/tuple/control_dependency_1Reshape_44/shape*
Tshape0*
_output_shapes

:??*
T0
c
Reshape_45/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
?

Reshape_45Reshape>gradients_4/vf/dense_1/BiasAdd_grad/tuple/control_dependency_1Reshape_45/shape*
Tshape0*
_output_shapes	
:?*
T0
c
Reshape_46/shapeConst*
dtype0*
_output_shapes
:*
valueB:
?????????
?

Reshape_46Reshape=gradients_4/vf/dense_2/MatMul_grad/tuple/control_dependency_1Reshape_46/shape*
_output_shapes	
:?*
Tshape0*
T0
c
Reshape_47/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
?

Reshape_47Reshape>gradients_4/vf/dense_2/BiasAdd_grad/tuple/control_dependency_1Reshape_47/shape*
T0*
Tshape0*
_output_shapes
:
c
Reshape_48/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
?

Reshape_48Reshape;gradients_4/vc/dense/MatMul_grad/tuple/control_dependency_1Reshape_48/shape*
Tshape0*
_output_shapes	
:?x*
T0
c
Reshape_49/shapeConst*
dtype0*
_output_shapes
:*
valueB:
?????????
?

Reshape_49Reshape<gradients_4/vc/dense/BiasAdd_grad/tuple/control_dependency_1Reshape_49/shape*
Tshape0*
_output_shapes	
:?*
T0
c
Reshape_50/shapeConst*
valueB:
?????????*
dtype0*
_output_shapes
:
?

Reshape_50Reshape=gradients_4/vc/dense_1/MatMul_grad/tuple/control_dependency_1Reshape_50/shape*
_output_shapes

:??*
T0*
Tshape0
c
Reshape_51/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
?

Reshape_51Reshape>gradients_4/vc/dense_1/BiasAdd_grad/tuple/control_dependency_1Reshape_51/shape*
_output_shapes	
:?*
T0*
Tshape0
c
Reshape_52/shapeConst*
dtype0*
valueB:
?????????*
_output_shapes
:
?

Reshape_52Reshape=gradients_4/vc/dense_2/MatMul_grad/tuple/control_dependency_1Reshape_52/shape*
Tshape0*
_output_shapes	
:?*
T0
c
Reshape_53/shapeConst*
valueB:
?????????*
_output_shapes
:*
dtype0
?

Reshape_53Reshape>gradients_4/vc/dense_2/BiasAdd_grad/tuple/control_dependency_1Reshape_53/shape*
T0*
_output_shapes
:*
Tshape0
O
concat_5/axisConst*
_output_shapes
: *
value	B : *
dtype0
?
concat_5ConcatV2
Reshape_42
Reshape_43
Reshape_44
Reshape_45
Reshape_46
Reshape_47
Reshape_48
Reshape_49
Reshape_50
Reshape_51
Reshape_52
Reshape_53concat_5/axis*
_output_shapes

:??	*

Tidx0*
T0*
N
j
PyFuncPyFuncconcat_5*
_output_shapes

:??	*
Tout
2*
Tin
2*
token
pyfunc_0
?
Const_7Const*
_output_shapes
:*E
value<B:"0 <                  <                 *
dtype0
S
split_1/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
?
split_1SplitVPyFuncConst_7split_1/split_dim*

Tlen0*
	num_split*h
_output_shapesV
T:?x:?:??:?:?::?x:?:??:?:?:*
T0
a
Reshape_54/shapeConst*
dtype0*
valueB"<      *
_output_shapes
:
h

Reshape_54Reshapesplit_1Reshape_54/shape*
T0*
Tshape0*
_output_shapes
:	<?
[
Reshape_55/shapeConst*
valueB:?*
dtype0*
_output_shapes
:
f

Reshape_55Reshape	split_1:1Reshape_55/shape*
Tshape0*
T0*
_output_shapes	
:?
a
Reshape_56/shapeConst*
dtype0*
_output_shapes
:*
valueB"      
k

Reshape_56Reshape	split_1:2Reshape_56/shape*
Tshape0* 
_output_shapes
:
??*
T0
[
Reshape_57/shapeConst*
valueB:?*
dtype0*
_output_shapes
:
f

Reshape_57Reshape	split_1:3Reshape_57/shape*
Tshape0*
_output_shapes	
:?*
T0
a
Reshape_58/shapeConst*
dtype0*
_output_shapes
:*
valueB"      
j

Reshape_58Reshape	split_1:4Reshape_58/shape*
T0*
_output_shapes
:	?*
Tshape0
Z
Reshape_59/shapeConst*
dtype0*
_output_shapes
:*
valueB:
e

Reshape_59Reshape	split_1:5Reshape_59/shape*
T0*
Tshape0*
_output_shapes
:
a
Reshape_60/shapeConst*
valueB"<      *
dtype0*
_output_shapes
:
j

Reshape_60Reshape	split_1:6Reshape_60/shape*
T0*
Tshape0*
_output_shapes
:	<?
[
Reshape_61/shapeConst*
_output_shapes
:*
dtype0*
valueB:?
f

Reshape_61Reshape	split_1:7Reshape_61/shape*
T0*
Tshape0*
_output_shapes	
:?
a
Reshape_62/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
k

Reshape_62Reshape	split_1:8Reshape_62/shape*
T0* 
_output_shapes
:
??*
Tshape0
[
Reshape_63/shapeConst*
dtype0*
valueB:?*
_output_shapes
:
f

Reshape_63Reshape	split_1:9Reshape_63/shape*
Tshape0*
_output_shapes	
:?*
T0
a
Reshape_64/shapeConst*
valueB"      *
_output_shapes
:*
dtype0
k

Reshape_64Reshape
split_1:10Reshape_64/shape*
_output_shapes
:	?*
T0*
Tshape0
Z
Reshape_65/shapeConst*
valueB:*
dtype0*
_output_shapes
:
f

Reshape_65Reshape
split_1:11Reshape_65/shape*
T0*
_output_shapes
:*
Tshape0
?
beta1_power/initial_valueConst*
_output_shapes
: * 
_class
loc:@vc/dense/bias*
valueB
 *fff?*
dtype0
?
beta1_power
VariableV2*
shared_name * 
_class
loc:@vc/dense/bias*
shape: *
	container *
dtype0*
_output_shapes
: 
?
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value* 
_class
loc:@vc/dense/bias*
_output_shapes
: *
T0*
use_locking(*
validate_shape(
l
beta1_power/readIdentitybeta1_power*
T0* 
_class
loc:@vc/dense/bias*
_output_shapes
: 
?
beta2_power/initial_valueConst*
dtype0*
valueB
 *w??* 
_class
loc:@vc/dense/bias*
_output_shapes
: 
?
beta2_power
VariableV2*
shared_name *
shape: *
dtype0*
_output_shapes
: *
	container * 
_class
loc:@vc/dense/bias
?
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
T0*
_output_shapes
: *
use_locking(* 
_class
loc:@vc/dense/bias*
validate_shape(
l
beta2_power/readIdentitybeta2_power* 
_class
loc:@vc/dense/bias*
_output_shapes
: *
T0
?
6vf/dense/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
valueB"<      *"
_class
loc:@vf/dense/kernel*
_output_shapes
:
?
,vf/dense/kernel/Adam/Initializer/zeros/ConstConst*"
_class
loc:@vf/dense/kernel*
_output_shapes
: *
valueB
 *    *
dtype0
?
&vf/dense/kernel/Adam/Initializer/zerosFill6vf/dense/kernel/Adam/Initializer/zeros/shape_as_tensor,vf/dense/kernel/Adam/Initializer/zeros/Const*

index_type0*
_output_shapes
:	<?*"
_class
loc:@vf/dense/kernel*
T0
?
vf/dense/kernel/Adam
VariableV2*
shape:	<?*
_output_shapes
:	<?*
	container *
dtype0*"
_class
loc:@vf/dense/kernel*
shared_name 
?
vf/dense/kernel/Adam/AssignAssignvf/dense/kernel/Adam&vf/dense/kernel/Adam/Initializer/zeros*
T0*
validate_shape(*
_output_shapes
:	<?*
use_locking(*"
_class
loc:@vf/dense/kernel
?
vf/dense/kernel/Adam/readIdentityvf/dense/kernel/Adam*
_output_shapes
:	<?*
T0*"
_class
loc:@vf/dense/kernel
?
8vf/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*"
_class
loc:@vf/dense/kernel*
_output_shapes
:*
valueB"<      *
dtype0
?
.vf/dense/kernel/Adam_1/Initializer/zeros/ConstConst*"
_class
loc:@vf/dense/kernel*
dtype0*
valueB
 *    *
_output_shapes
: 
?
(vf/dense/kernel/Adam_1/Initializer/zerosFill8vf/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensor.vf/dense/kernel/Adam_1/Initializer/zeros/Const*"
_class
loc:@vf/dense/kernel*
_output_shapes
:	<?*
T0*

index_type0
?
vf/dense/kernel/Adam_1
VariableV2*"
_class
loc:@vf/dense/kernel*
_output_shapes
:	<?*
dtype0*
	container *
shared_name *
shape:	<?
?
vf/dense/kernel/Adam_1/AssignAssignvf/dense/kernel/Adam_1(vf/dense/kernel/Adam_1/Initializer/zeros*"
_class
loc:@vf/dense/kernel*
_output_shapes
:	<?*
validate_shape(*
use_locking(*
T0
?
vf/dense/kernel/Adam_1/readIdentityvf/dense/kernel/Adam_1*
_output_shapes
:	<?*
T0*"
_class
loc:@vf/dense/kernel
?
$vf/dense/bias/Adam/Initializer/zerosConst*
valueB?*    *
dtype0* 
_class
loc:@vf/dense/bias*
_output_shapes	
:?
?
vf/dense/bias/Adam
VariableV2*
shared_name *
dtype0*
_output_shapes	
:?*
shape:?* 
_class
loc:@vf/dense/bias*
	container 
?
vf/dense/bias/Adam/AssignAssignvf/dense/bias/Adam$vf/dense/bias/Adam/Initializer/zeros*
T0*
_output_shapes	
:?* 
_class
loc:@vf/dense/bias*
use_locking(*
validate_shape(

vf/dense/bias/Adam/readIdentityvf/dense/bias/Adam* 
_class
loc:@vf/dense/bias*
_output_shapes	
:?*
T0
?
&vf/dense/bias/Adam_1/Initializer/zerosConst*
dtype0* 
_class
loc:@vf/dense/bias*
_output_shapes	
:?*
valueB?*    
?
vf/dense/bias/Adam_1
VariableV2*
shared_name * 
_class
loc:@vf/dense/bias*
shape:?*
_output_shapes	
:?*
	container *
dtype0
?
vf/dense/bias/Adam_1/AssignAssignvf/dense/bias/Adam_1&vf/dense/bias/Adam_1/Initializer/zeros*
_output_shapes	
:?*
validate_shape(* 
_class
loc:@vf/dense/bias*
use_locking(*
T0
?
vf/dense/bias/Adam_1/readIdentityvf/dense/bias/Adam_1* 
_class
loc:@vf/dense/bias*
_output_shapes	
:?*
T0
?
8vf/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*$
_class
loc:@vf/dense_1/kernel*
dtype0*
_output_shapes
:*
valueB"      
?
.vf/dense_1/kernel/Adam/Initializer/zeros/ConstConst*
dtype0*$
_class
loc:@vf/dense_1/kernel*
valueB
 *    *
_output_shapes
: 
?
(vf/dense_1/kernel/Adam/Initializer/zerosFill8vf/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensor.vf/dense_1/kernel/Adam/Initializer/zeros/Const*$
_class
loc:@vf/dense_1/kernel*
T0*

index_type0* 
_output_shapes
:
??
?
vf/dense_1/kernel/Adam
VariableV2*$
_class
loc:@vf/dense_1/kernel* 
_output_shapes
:
??*
dtype0*
shape:
??*
shared_name *
	container 
?
vf/dense_1/kernel/Adam/AssignAssignvf/dense_1/kernel/Adam(vf/dense_1/kernel/Adam/Initializer/zeros*$
_class
loc:@vf/dense_1/kernel*
use_locking(*
validate_shape(*
T0* 
_output_shapes
:
??
?
vf/dense_1/kernel/Adam/readIdentityvf/dense_1/kernel/Adam*
T0* 
_output_shapes
:
??*$
_class
loc:@vf/dense_1/kernel
?
:vf/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*$
_class
loc:@vf/dense_1/kernel*
_output_shapes
:*
valueB"      
?
0vf/dense_1/kernel/Adam_1/Initializer/zeros/ConstConst*$
_class
loc:@vf/dense_1/kernel*
_output_shapes
: *
valueB
 *    *
dtype0
?
*vf/dense_1/kernel/Adam_1/Initializer/zerosFill:vf/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensor0vf/dense_1/kernel/Adam_1/Initializer/zeros/Const*$
_class
loc:@vf/dense_1/kernel*

index_type0* 
_output_shapes
:
??*
T0
?
vf/dense_1/kernel/Adam_1
VariableV2*
shape:
??*
	container *
shared_name *
dtype0* 
_output_shapes
:
??*$
_class
loc:@vf/dense_1/kernel
?
vf/dense_1/kernel/Adam_1/AssignAssignvf/dense_1/kernel/Adam_1*vf/dense_1/kernel/Adam_1/Initializer/zeros*
T0*
validate_shape(*$
_class
loc:@vf/dense_1/kernel*
use_locking(* 
_output_shapes
:
??
?
vf/dense_1/kernel/Adam_1/readIdentityvf/dense_1/kernel/Adam_1*$
_class
loc:@vf/dense_1/kernel*
T0* 
_output_shapes
:
??
?
&vf/dense_1/bias/Adam/Initializer/zerosConst*"
_class
loc:@vf/dense_1/bias*
valueB?*    *
dtype0*
_output_shapes	
:?
?
vf/dense_1/bias/Adam
VariableV2*
shape:?*"
_class
loc:@vf/dense_1/bias*
dtype0*
	container *
_output_shapes	
:?*
shared_name 
?
vf/dense_1/bias/Adam/AssignAssignvf/dense_1/bias/Adam&vf/dense_1/bias/Adam/Initializer/zeros*
use_locking(*
_output_shapes	
:?*
validate_shape(*"
_class
loc:@vf/dense_1/bias*
T0
?
vf/dense_1/bias/Adam/readIdentityvf/dense_1/bias/Adam*
T0*
_output_shapes	
:?*"
_class
loc:@vf/dense_1/bias
?
(vf/dense_1/bias/Adam_1/Initializer/zerosConst*
valueB?*    *
dtype0*
_output_shapes	
:?*"
_class
loc:@vf/dense_1/bias
?
vf/dense_1/bias/Adam_1
VariableV2*
shape:?*
dtype0*"
_class
loc:@vf/dense_1/bias*
_output_shapes	
:?*
	container *
shared_name 
?
vf/dense_1/bias/Adam_1/AssignAssignvf/dense_1/bias/Adam_1(vf/dense_1/bias/Adam_1/Initializer/zeros*
use_locking(*"
_class
loc:@vf/dense_1/bias*
validate_shape(*
T0*
_output_shapes	
:?
?
vf/dense_1/bias/Adam_1/readIdentityvf/dense_1/bias/Adam_1*"
_class
loc:@vf/dense_1/bias*
_output_shapes	
:?*
T0
?
(vf/dense_2/kernel/Adam/Initializer/zerosConst*
_output_shapes
:	?*
dtype0*$
_class
loc:@vf/dense_2/kernel*
valueB	?*    
?
vf/dense_2/kernel/Adam
VariableV2*
shared_name *
_output_shapes
:	?*
	container *$
_class
loc:@vf/dense_2/kernel*
dtype0*
shape:	?
?
vf/dense_2/kernel/Adam/AssignAssignvf/dense_2/kernel/Adam(vf/dense_2/kernel/Adam/Initializer/zeros*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	?*
validate_shape(*
use_locking(*
T0
?
vf/dense_2/kernel/Adam/readIdentityvf/dense_2/kernel/Adam*
T0*
_output_shapes
:	?*$
_class
loc:@vf/dense_2/kernel
?
*vf/dense_2/kernel/Adam_1/Initializer/zerosConst*
valueB	?*    *$
_class
loc:@vf/dense_2/kernel*
dtype0*
_output_shapes
:	?
?
vf/dense_2/kernel/Adam_1
VariableV2*
	container *
_output_shapes
:	?*$
_class
loc:@vf/dense_2/kernel*
dtype0*
shared_name *
shape:	?
?
vf/dense_2/kernel/Adam_1/AssignAssignvf/dense_2/kernel/Adam_1*vf/dense_2/kernel/Adam_1/Initializer/zeros*
T0*
_output_shapes
:	?*
use_locking(*
validate_shape(*$
_class
loc:@vf/dense_2/kernel
?
vf/dense_2/kernel/Adam_1/readIdentityvf/dense_2/kernel/Adam_1*$
_class
loc:@vf/dense_2/kernel*
T0*
_output_shapes
:	?
?
&vf/dense_2/bias/Adam/Initializer/zerosConst*"
_class
loc:@vf/dense_2/bias*
dtype0*
_output_shapes
:*
valueB*    
?
vf/dense_2/bias/Adam
VariableV2*
shared_name *
dtype0*"
_class
loc:@vf/dense_2/bias*
_output_shapes
:*
shape:*
	container 
?
vf/dense_2/bias/Adam/AssignAssignvf/dense_2/bias/Adam&vf/dense_2/bias/Adam/Initializer/zeros*
_output_shapes
:*
use_locking(*
validate_shape(*"
_class
loc:@vf/dense_2/bias*
T0
?
vf/dense_2/bias/Adam/readIdentityvf/dense_2/bias/Adam*
_output_shapes
:*"
_class
loc:@vf/dense_2/bias*
T0
?
(vf/dense_2/bias/Adam_1/Initializer/zerosConst*"
_class
loc:@vf/dense_2/bias*
valueB*    *
_output_shapes
:*
dtype0
?
vf/dense_2/bias/Adam_1
VariableV2*"
_class
loc:@vf/dense_2/bias*
	container *
shared_name *
shape:*
_output_shapes
:*
dtype0
?
vf/dense_2/bias/Adam_1/AssignAssignvf/dense_2/bias/Adam_1(vf/dense_2/bias/Adam_1/Initializer/zeros*"
_class
loc:@vf/dense_2/bias*
validate_shape(*
use_locking(*
_output_shapes
:*
T0
?
vf/dense_2/bias/Adam_1/readIdentityvf/dense_2/bias/Adam_1*"
_class
loc:@vf/dense_2/bias*
T0*
_output_shapes
:
?
6vc/dense/kernel/Adam/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB"<      *"
_class
loc:@vc/dense/kernel
?
,vc/dense/kernel/Adam/Initializer/zeros/ConstConst*"
_class
loc:@vc/dense/kernel*
_output_shapes
: *
dtype0*
valueB
 *    
?
&vc/dense/kernel/Adam/Initializer/zerosFill6vc/dense/kernel/Adam/Initializer/zeros/shape_as_tensor,vc/dense/kernel/Adam/Initializer/zeros/Const*
_output_shapes
:	<?*"
_class
loc:@vc/dense/kernel*
T0*

index_type0
?
vc/dense/kernel/Adam
VariableV2*"
_class
loc:@vc/dense/kernel*
shared_name *
	container *
dtype0*
_output_shapes
:	<?*
shape:	<?
?
vc/dense/kernel/Adam/AssignAssignvc/dense/kernel/Adam&vc/dense/kernel/Adam/Initializer/zeros*"
_class
loc:@vc/dense/kernel*
use_locking(*
T0*
_output_shapes
:	<?*
validate_shape(
?
vc/dense/kernel/Adam/readIdentityvc/dense/kernel/Adam*
T0*"
_class
loc:@vc/dense/kernel*
_output_shapes
:	<?
?
8vc/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
dtype0*
valueB"<      *
_output_shapes
:*"
_class
loc:@vc/dense/kernel
?
.vc/dense/kernel/Adam_1/Initializer/zeros/ConstConst*"
_class
loc:@vc/dense/kernel*
valueB
 *    *
_output_shapes
: *
dtype0
?
(vc/dense/kernel/Adam_1/Initializer/zerosFill8vc/dense/kernel/Adam_1/Initializer/zeros/shape_as_tensor.vc/dense/kernel/Adam_1/Initializer/zeros/Const*
T0*
_output_shapes
:	<?*"
_class
loc:@vc/dense/kernel*

index_type0
?
vc/dense/kernel/Adam_1
VariableV2*
	container *
shape:	<?*
dtype0*"
_class
loc:@vc/dense/kernel*
_output_shapes
:	<?*
shared_name 
?
vc/dense/kernel/Adam_1/AssignAssignvc/dense/kernel/Adam_1(vc/dense/kernel/Adam_1/Initializer/zeros*
use_locking(*
validate_shape(*
_output_shapes
:	<?*
T0*"
_class
loc:@vc/dense/kernel
?
vc/dense/kernel/Adam_1/readIdentityvc/dense/kernel/Adam_1*
T0*
_output_shapes
:	<?*"
_class
loc:@vc/dense/kernel
?
$vc/dense/bias/Adam/Initializer/zerosConst* 
_class
loc:@vc/dense/bias*
valueB?*    *
dtype0*
_output_shapes	
:?
?
vc/dense/bias/Adam
VariableV2*
dtype0*
	container *
shape:?* 
_class
loc:@vc/dense/bias*
_output_shapes	
:?*
shared_name 
?
vc/dense/bias/Adam/AssignAssignvc/dense/bias/Adam$vc/dense/bias/Adam/Initializer/zeros*
_output_shapes	
:?*
use_locking(*
T0* 
_class
loc:@vc/dense/bias*
validate_shape(

vc/dense/bias/Adam/readIdentityvc/dense/bias/Adam*
_output_shapes	
:?* 
_class
loc:@vc/dense/bias*
T0
?
&vc/dense/bias/Adam_1/Initializer/zerosConst*
dtype0* 
_class
loc:@vc/dense/bias*
_output_shapes	
:?*
valueB?*    
?
vc/dense/bias/Adam_1
VariableV2*
shape:?* 
_class
loc:@vc/dense/bias*
dtype0*
	container *
shared_name *
_output_shapes	
:?
?
vc/dense/bias/Adam_1/AssignAssignvc/dense/bias/Adam_1&vc/dense/bias/Adam_1/Initializer/zeros*
T0*
_output_shapes	
:?*
validate_shape(* 
_class
loc:@vc/dense/bias*
use_locking(
?
vc/dense/bias/Adam_1/readIdentityvc/dense/bias/Adam_1*
_output_shapes	
:?*
T0* 
_class
loc:@vc/dense/bias
?
8vc/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*$
_class
loc:@vc/dense_1/kernel*
_output_shapes
:*
dtype0*
valueB"      
?
.vc/dense_1/kernel/Adam/Initializer/zeros/ConstConst*
valueB
 *    *$
_class
loc:@vc/dense_1/kernel*
_output_shapes
: *
dtype0
?
(vc/dense_1/kernel/Adam/Initializer/zerosFill8vc/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensor.vc/dense_1/kernel/Adam/Initializer/zeros/Const*$
_class
loc:@vc/dense_1/kernel*

index_type0* 
_output_shapes
:
??*
T0
?
vc/dense_1/kernel/Adam
VariableV2* 
_output_shapes
:
??*
shared_name *$
_class
loc:@vc/dense_1/kernel*
shape:
??*
dtype0*
	container 
?
vc/dense_1/kernel/Adam/AssignAssignvc/dense_1/kernel/Adam(vc/dense_1/kernel/Adam/Initializer/zeros*
use_locking(*$
_class
loc:@vc/dense_1/kernel*
T0* 
_output_shapes
:
??*
validate_shape(
?
vc/dense_1/kernel/Adam/readIdentityvc/dense_1/kernel/Adam*$
_class
loc:@vc/dense_1/kernel*
T0* 
_output_shapes
:
??
?
:vc/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*
_output_shapes
:*
valueB"      *
dtype0*$
_class
loc:@vc/dense_1/kernel
?
0vc/dense_1/kernel/Adam_1/Initializer/zeros/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    *$
_class
loc:@vc/dense_1/kernel
?
*vc/dense_1/kernel/Adam_1/Initializer/zerosFill:vc/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensor0vc/dense_1/kernel/Adam_1/Initializer/zeros/Const*$
_class
loc:@vc/dense_1/kernel*

index_type0*
T0* 
_output_shapes
:
??
?
vc/dense_1/kernel/Adam_1
VariableV2*
shared_name *$
_class
loc:@vc/dense_1/kernel*
dtype0*
shape:
??* 
_output_shapes
:
??*
	container 
?
vc/dense_1/kernel/Adam_1/AssignAssignvc/dense_1/kernel/Adam_1*vc/dense_1/kernel/Adam_1/Initializer/zeros*
T0*
use_locking(*
validate_shape(* 
_output_shapes
:
??*$
_class
loc:@vc/dense_1/kernel
?
vc/dense_1/kernel/Adam_1/readIdentityvc/dense_1/kernel/Adam_1* 
_output_shapes
:
??*
T0*$
_class
loc:@vc/dense_1/kernel
?
&vc/dense_1/bias/Adam/Initializer/zerosConst*
dtype0*"
_class
loc:@vc/dense_1/bias*
valueB?*    *
_output_shapes	
:?
?
vc/dense_1/bias/Adam
VariableV2*"
_class
loc:@vc/dense_1/bias*
_output_shapes	
:?*
shared_name *
	container *
shape:?*
dtype0
?
vc/dense_1/bias/Adam/AssignAssignvc/dense_1/bias/Adam&vc/dense_1/bias/Adam/Initializer/zeros*
validate_shape(*"
_class
loc:@vc/dense_1/bias*
use_locking(*
T0*
_output_shapes	
:?
?
vc/dense_1/bias/Adam/readIdentityvc/dense_1/bias/Adam*
_output_shapes	
:?*
T0*"
_class
loc:@vc/dense_1/bias
?
(vc/dense_1/bias/Adam_1/Initializer/zerosConst*
valueB?*    *
_output_shapes	
:?*
dtype0*"
_class
loc:@vc/dense_1/bias
?
vc/dense_1/bias/Adam_1
VariableV2*
_output_shapes	
:?*
shared_name *"
_class
loc:@vc/dense_1/bias*
dtype0*
	container *
shape:?
?
vc/dense_1/bias/Adam_1/AssignAssignvc/dense_1/bias/Adam_1(vc/dense_1/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*
_output_shapes	
:?*"
_class
loc:@vc/dense_1/bias*
validate_shape(
?
vc/dense_1/bias/Adam_1/readIdentityvc/dense_1/bias/Adam_1*
T0*"
_class
loc:@vc/dense_1/bias*
_output_shapes	
:?
?
(vc/dense_2/kernel/Adam/Initializer/zerosConst*$
_class
loc:@vc/dense_2/kernel*
valueB	?*    *
_output_shapes
:	?*
dtype0
?
vc/dense_2/kernel/Adam
VariableV2*
	container *$
_class
loc:@vc/dense_2/kernel*
_output_shapes
:	?*
dtype0*
shared_name *
shape:	?
?
vc/dense_2/kernel/Adam/AssignAssignvc/dense_2/kernel/Adam(vc/dense_2/kernel/Adam/Initializer/zeros*
_output_shapes
:	?*
use_locking(*
validate_shape(*
T0*$
_class
loc:@vc/dense_2/kernel
?
vc/dense_2/kernel/Adam/readIdentityvc/dense_2/kernel/Adam*
T0*
_output_shapes
:	?*$
_class
loc:@vc/dense_2/kernel
?
*vc/dense_2/kernel/Adam_1/Initializer/zerosConst*
valueB	?*    *$
_class
loc:@vc/dense_2/kernel*
dtype0*
_output_shapes
:	?
?
vc/dense_2/kernel/Adam_1
VariableV2*
shape:	?*
dtype0*
shared_name *
_output_shapes
:	?*
	container *$
_class
loc:@vc/dense_2/kernel
?
vc/dense_2/kernel/Adam_1/AssignAssignvc/dense_2/kernel/Adam_1*vc/dense_2/kernel/Adam_1/Initializer/zeros*
T0*
_output_shapes
:	?*$
_class
loc:@vc/dense_2/kernel*
validate_shape(*
use_locking(
?
vc/dense_2/kernel/Adam_1/readIdentityvc/dense_2/kernel/Adam_1*
T0*
_output_shapes
:	?*$
_class
loc:@vc/dense_2/kernel
?
&vc/dense_2/bias/Adam/Initializer/zerosConst*
valueB*    *
_output_shapes
:*"
_class
loc:@vc/dense_2/bias*
dtype0
?
vc/dense_2/bias/Adam
VariableV2*
	container *"
_class
loc:@vc/dense_2/bias*
_output_shapes
:*
shape:*
dtype0*
shared_name 
?
vc/dense_2/bias/Adam/AssignAssignvc/dense_2/bias/Adam&vc/dense_2/bias/Adam/Initializer/zeros*
validate_shape(*
T0*
use_locking(*
_output_shapes
:*"
_class
loc:@vc/dense_2/bias
?
vc/dense_2/bias/Adam/readIdentityvc/dense_2/bias/Adam*"
_class
loc:@vc/dense_2/bias*
T0*
_output_shapes
:
?
(vc/dense_2/bias/Adam_1/Initializer/zerosConst*
valueB*    *
_output_shapes
:*
dtype0*"
_class
loc:@vc/dense_2/bias
?
vc/dense_2/bias/Adam_1
VariableV2*
	container *
shared_name *"
_class
loc:@vc/dense_2/bias*
_output_shapes
:*
shape:*
dtype0
?
vc/dense_2/bias/Adam_1/AssignAssignvc/dense_2/bias/Adam_1(vc/dense_2/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*
validate_shape(*"
_class
loc:@vc/dense_2/bias*
_output_shapes
:
?
vc/dense_2/bias/Adam_1/readIdentityvc/dense_2/bias/Adam_1*
_output_shapes
:*"
_class
loc:@vc/dense_2/bias*
T0
W
Adam/learning_rateConst*
valueB
 *o?:*
_output_shapes
: *
dtype0
O

Adam/beta1Const*
dtype0*
_output_shapes
: *
valueB
 *fff?
O

Adam/beta2Const*
valueB
 *w??*
_output_shapes
: *
dtype0
Q
Adam/epsilonConst*
valueB
 *w?+2*
dtype0*
_output_shapes
: 
?
%Adam/update_vf/dense/kernel/ApplyAdam	ApplyAdamvf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon
Reshape_54*
use_nesterov( *
_output_shapes
:	<?*"
_class
loc:@vf/dense/kernel*
T0*
use_locking( 
?
#Adam/update_vf/dense/bias/ApplyAdam	ApplyAdamvf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon
Reshape_55*
T0*
_output_shapes	
:?*
use_nesterov( *
use_locking( * 
_class
loc:@vf/dense/bias
?
'Adam/update_vf/dense_1/kernel/ApplyAdam	ApplyAdamvf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon
Reshape_56*
use_locking( *$
_class
loc:@vf/dense_1/kernel*
use_nesterov( * 
_output_shapes
:
??*
T0
?
%Adam/update_vf/dense_1/bias/ApplyAdam	ApplyAdamvf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon
Reshape_57*
_output_shapes	
:?*
use_locking( *"
_class
loc:@vf/dense_1/bias*
use_nesterov( *
T0
?
'Adam/update_vf/dense_2/kernel/ApplyAdam	ApplyAdamvf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon
Reshape_58*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	?*
use_nesterov( *
use_locking( *
T0
?
%Adam/update_vf/dense_2/bias/ApplyAdam	ApplyAdamvf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon
Reshape_59*
_output_shapes
:*"
_class
loc:@vf/dense_2/bias*
use_nesterov( *
T0*
use_locking( 
?
%Adam/update_vc/dense/kernel/ApplyAdam	ApplyAdamvc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon
Reshape_60*
use_nesterov( *"
_class
loc:@vc/dense/kernel*
_output_shapes
:	<?*
use_locking( *
T0
?
#Adam/update_vc/dense/bias/ApplyAdam	ApplyAdamvc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon
Reshape_61*
use_nesterov( * 
_class
loc:@vc/dense/bias*
use_locking( *
T0*
_output_shapes	
:?
?
'Adam/update_vc/dense_1/kernel/ApplyAdam	ApplyAdamvc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon
Reshape_62*
use_nesterov( * 
_output_shapes
:
??*
use_locking( *$
_class
loc:@vc/dense_1/kernel*
T0
?
%Adam/update_vc/dense_1/bias/ApplyAdam	ApplyAdamvc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon
Reshape_63*
use_locking( *"
_class
loc:@vc/dense_1/bias*
_output_shapes	
:?*
use_nesterov( *
T0
?
'Adam/update_vc/dense_2/kernel/ApplyAdam	ApplyAdamvc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon
Reshape_64*$
_class
loc:@vc/dense_2/kernel*
_output_shapes
:	?*
T0*
use_locking( *
use_nesterov( 
?
%Adam/update_vc/dense_2/bias/ApplyAdam	ApplyAdamvc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilon
Reshape_65*
use_locking( *
T0*
use_nesterov( *"
_class
loc:@vc/dense_2/bias*
_output_shapes
:
?
Adam/mulMulbeta1_power/read
Adam/beta1$^Adam/update_vc/dense/bias/ApplyAdam&^Adam/update_vc/dense/kernel/ApplyAdam&^Adam/update_vc/dense_1/bias/ApplyAdam(^Adam/update_vc/dense_1/kernel/ApplyAdam&^Adam/update_vc/dense_2/bias/ApplyAdam(^Adam/update_vc/dense_2/kernel/ApplyAdam$^Adam/update_vf/dense/bias/ApplyAdam&^Adam/update_vf/dense/kernel/ApplyAdam&^Adam/update_vf/dense_1/bias/ApplyAdam(^Adam/update_vf/dense_1/kernel/ApplyAdam&^Adam/update_vf/dense_2/bias/ApplyAdam(^Adam/update_vf/dense_2/kernel/ApplyAdam* 
_class
loc:@vc/dense/bias*
T0*
_output_shapes
: 
?
Adam/AssignAssignbeta1_powerAdam/mul*
_output_shapes
: *
use_locking( * 
_class
loc:@vc/dense/bias*
validate_shape(*
T0
?

Adam/mul_1Mulbeta2_power/read
Adam/beta2$^Adam/update_vc/dense/bias/ApplyAdam&^Adam/update_vc/dense/kernel/ApplyAdam&^Adam/update_vc/dense_1/bias/ApplyAdam(^Adam/update_vc/dense_1/kernel/ApplyAdam&^Adam/update_vc/dense_2/bias/ApplyAdam(^Adam/update_vc/dense_2/kernel/ApplyAdam$^Adam/update_vf/dense/bias/ApplyAdam&^Adam/update_vf/dense/kernel/ApplyAdam&^Adam/update_vf/dense_1/bias/ApplyAdam(^Adam/update_vf/dense_1/kernel/ApplyAdam&^Adam/update_vf/dense_2/bias/ApplyAdam(^Adam/update_vf/dense_2/kernel/ApplyAdam*
_output_shapes
: * 
_class
loc:@vc/dense/bias*
T0
?
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
T0*
validate_shape(* 
_class
loc:@vc/dense/bias*
use_locking( *
_output_shapes
: 
?
AdamNoOp^Adam/Assign^Adam/Assign_1$^Adam/update_vc/dense/bias/ApplyAdam&^Adam/update_vc/dense/kernel/ApplyAdam&^Adam/update_vc/dense_1/bias/ApplyAdam(^Adam/update_vc/dense_1/kernel/ApplyAdam&^Adam/update_vc/dense_2/bias/ApplyAdam(^Adam/update_vc/dense_2/kernel/ApplyAdam$^Adam/update_vf/dense/bias/ApplyAdam&^Adam/update_vf/dense/kernel/ApplyAdam&^Adam/update_vf/dense_1/bias/ApplyAdam(^Adam/update_vf/dense_1/kernel/ApplyAdam&^Adam/update_vf/dense_2/bias/ApplyAdam(^Adam/update_vf/dense_2/kernel/ApplyAdam
j
Reshape_66/shapeConst^Adam*
dtype0*
valueB:
?????????*
_output_shapes
:
q

Reshape_66Reshapevf/dense/kernel/readReshape_66/shape*
_output_shapes	
:?x*
Tshape0*
T0
j
Reshape_67/shapeConst^Adam*
dtype0*
valueB:
?????????*
_output_shapes
:
o

Reshape_67Reshapevf/dense/bias/readReshape_67/shape*
_output_shapes	
:?*
Tshape0*
T0
j
Reshape_68/shapeConst^Adam*
dtype0*
_output_shapes
:*
valueB:
?????????
t

Reshape_68Reshapevf/dense_1/kernel/readReshape_68/shape*
T0*
_output_shapes

:??*
Tshape0
j
Reshape_69/shapeConst^Adam*
_output_shapes
:*
dtype0*
valueB:
?????????
q

Reshape_69Reshapevf/dense_1/bias/readReshape_69/shape*
T0*
_output_shapes	
:?*
Tshape0
j
Reshape_70/shapeConst^Adam*
_output_shapes
:*
dtype0*
valueB:
?????????
s

Reshape_70Reshapevf/dense_2/kernel/readReshape_70/shape*
_output_shapes	
:?*
Tshape0*
T0
j
Reshape_71/shapeConst^Adam*
dtype0*
_output_shapes
:*
valueB:
?????????
p

Reshape_71Reshapevf/dense_2/bias/readReshape_71/shape*
_output_shapes
:*
Tshape0*
T0
j
Reshape_72/shapeConst^Adam*
dtype0*
valueB:
?????????*
_output_shapes
:
q

Reshape_72Reshapevc/dense/kernel/readReshape_72/shape*
Tshape0*
T0*
_output_shapes	
:?x
j
Reshape_73/shapeConst^Adam*
_output_shapes
:*
valueB:
?????????*
dtype0
o

Reshape_73Reshapevc/dense/bias/readReshape_73/shape*
_output_shapes	
:?*
T0*
Tshape0
j
Reshape_74/shapeConst^Adam*
valueB:
?????????*
_output_shapes
:*
dtype0
t

Reshape_74Reshapevc/dense_1/kernel/readReshape_74/shape*
Tshape0*
_output_shapes

:??*
T0
j
Reshape_75/shapeConst^Adam*
_output_shapes
:*
valueB:
?????????*
dtype0
q

Reshape_75Reshapevc/dense_1/bias/readReshape_75/shape*
Tshape0*
T0*
_output_shapes	
:?
j
Reshape_76/shapeConst^Adam*
valueB:
?????????*
_output_shapes
:*
dtype0
s

Reshape_76Reshapevc/dense_2/kernel/readReshape_76/shape*
_output_shapes	
:?*
T0*
Tshape0
j
Reshape_77/shapeConst^Adam*
valueB:
?????????*
_output_shapes
:*
dtype0
p

Reshape_77Reshapevc/dense_2/bias/readReshape_77/shape*
_output_shapes
:*
T0*
Tshape0
V
concat_6/axisConst^Adam*
dtype0*
_output_shapes
: *
value	B : 
?
concat_6ConcatV2
Reshape_66
Reshape_67
Reshape_68
Reshape_69
Reshape_70
Reshape_71
Reshape_72
Reshape_73
Reshape_74
Reshape_75
Reshape_76
Reshape_77concat_6/axis*

Tidx0*
N*
_output_shapes

:??	*
T0
h
PyFunc_1PyFuncconcat_6*
Tin
2*
Tout
2*
_output_shapes
:*
token
pyfunc_1
?
Const_8Const^Adam*
_output_shapes
:*E
value<B:"0 <                  <                 *
dtype0
Z
split_2/split_dimConst^Adam*
_output_shapes
: *
value	B : *
dtype0
?
split_2SplitVPyFunc_1Const_8split_2/split_dim*D
_output_shapes2
0::::::::::::*

Tlen0*
	num_split*
T0
h
Reshape_78/shapeConst^Adam*
dtype0*
valueB"<      *
_output_shapes
:
h

Reshape_78Reshapesplit_2Reshape_78/shape*
T0*
Tshape0*
_output_shapes
:	<?
b
Reshape_79/shapeConst^Adam*
dtype0*
valueB:?*
_output_shapes
:
f

Reshape_79Reshape	split_2:1Reshape_79/shape*
Tshape0*
_output_shapes	
:?*
T0
h
Reshape_80/shapeConst^Adam*
valueB"      *
dtype0*
_output_shapes
:
k

Reshape_80Reshape	split_2:2Reshape_80/shape*
T0* 
_output_shapes
:
??*
Tshape0
b
Reshape_81/shapeConst^Adam*
valueB:?*
dtype0*
_output_shapes
:
f

Reshape_81Reshape	split_2:3Reshape_81/shape*
_output_shapes	
:?*
T0*
Tshape0
h
Reshape_82/shapeConst^Adam*
valueB"      *
dtype0*
_output_shapes
:
j

Reshape_82Reshape	split_2:4Reshape_82/shape*
_output_shapes
:	?*
T0*
Tshape0
a
Reshape_83/shapeConst^Adam*
valueB:*
_output_shapes
:*
dtype0
e

Reshape_83Reshape	split_2:5Reshape_83/shape*
Tshape0*
_output_shapes
:*
T0
h
Reshape_84/shapeConst^Adam*
_output_shapes
:*
valueB"<      *
dtype0
j

Reshape_84Reshape	split_2:6Reshape_84/shape*
_output_shapes
:	<?*
Tshape0*
T0
b
Reshape_85/shapeConst^Adam*
dtype0*
_output_shapes
:*
valueB:?
f

Reshape_85Reshape	split_2:7Reshape_85/shape*
_output_shapes	
:?*
Tshape0*
T0
h
Reshape_86/shapeConst^Adam*
valueB"      *
_output_shapes
:*
dtype0
k

Reshape_86Reshape	split_2:8Reshape_86/shape*
T0* 
_output_shapes
:
??*
Tshape0
b
Reshape_87/shapeConst^Adam*
_output_shapes
:*
dtype0*
valueB:?
f

Reshape_87Reshape	split_2:9Reshape_87/shape*
T0*
_output_shapes	
:?*
Tshape0
h
Reshape_88/shapeConst^Adam*
valueB"      *
dtype0*
_output_shapes
:
k

Reshape_88Reshape
split_2:10Reshape_88/shape*
_output_shapes
:	?*
T0*
Tshape0
a
Reshape_89/shapeConst^Adam*
valueB:*
_output_shapes
:*
dtype0
f

Reshape_89Reshape
split_2:11Reshape_89/shape*
Tshape0*
_output_shapes
:*
T0
?
Assign_7Assignvf/dense/kernel
Reshape_78*
validate_shape(*
use_locking(*
T0*
_output_shapes
:	<?*"
_class
loc:@vf/dense/kernel
?
Assign_8Assignvf/dense/bias
Reshape_79*
_output_shapes	
:?*
validate_shape(* 
_class
loc:@vf/dense/bias*
T0*
use_locking(
?
Assign_9Assignvf/dense_1/kernel
Reshape_80* 
_output_shapes
:
??*
validate_shape(*
T0*
use_locking(*$
_class
loc:@vf/dense_1/kernel
?
	Assign_10Assignvf/dense_1/bias
Reshape_81*
_output_shapes	
:?*
validate_shape(*
use_locking(*"
_class
loc:@vf/dense_1/bias*
T0
?
	Assign_11Assignvf/dense_2/kernel
Reshape_82*
use_locking(*
T0*
validate_shape(*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	?
?
	Assign_12Assignvf/dense_2/bias
Reshape_83*"
_class
loc:@vf/dense_2/bias*
_output_shapes
:*
T0*
validate_shape(*
use_locking(
?
	Assign_13Assignvc/dense/kernel
Reshape_84*
_output_shapes
:	<?*
validate_shape(*
use_locking(*
T0*"
_class
loc:@vc/dense/kernel
?
	Assign_14Assignvc/dense/bias
Reshape_85*
validate_shape(* 
_class
loc:@vc/dense/bias*
T0*
use_locking(*
_output_shapes	
:?
?
	Assign_15Assignvc/dense_1/kernel
Reshape_86* 
_output_shapes
:
??*
use_locking(*$
_class
loc:@vc/dense_1/kernel*
validate_shape(*
T0
?
	Assign_16Assignvc/dense_1/bias
Reshape_87*
validate_shape(*
use_locking(*
_output_shapes	
:?*
T0*"
_class
loc:@vc/dense_1/bias
?
	Assign_17Assignvc/dense_2/kernel
Reshape_88*$
_class
loc:@vc/dense_2/kernel*
use_locking(*
_output_shapes
:	?*
T0*
validate_shape(
?
	Assign_18Assignvc/dense_2/bias
Reshape_89*
T0*
use_locking(*"
_class
loc:@vc/dense_2/bias*
validate_shape(*
_output_shapes
:
?
group_deps_1NoOp^Adam
^Assign_10
^Assign_11
^Assign_12
^Assign_13
^Assign_14
^Assign_15
^Assign_16
^Assign_17
^Assign_18	^Assign_7	^Assign_8	^Assign_9
*
group_deps_2NoOp^Adam^group_deps_1
?

initNoOp^beta1_power/Assign^beta2_power/Assign^pi/dense/bias/Assign^pi/dense/kernel/Assign^pi/dense_1/bias/Assign^pi/dense_1/kernel/Assign^pi/dense_2/bias/Assign^pi/dense_2/kernel/Assign^pi/log_std/Assign^vc/dense/bias/Adam/Assign^vc/dense/bias/Adam_1/Assign^vc/dense/bias/Assign^vc/dense/kernel/Adam/Assign^vc/dense/kernel/Adam_1/Assign^vc/dense/kernel/Assign^vc/dense_1/bias/Adam/Assign^vc/dense_1/bias/Adam_1/Assign^vc/dense_1/bias/Assign^vc/dense_1/kernel/Adam/Assign ^vc/dense_1/kernel/Adam_1/Assign^vc/dense_1/kernel/Assign^vc/dense_2/bias/Adam/Assign^vc/dense_2/bias/Adam_1/Assign^vc/dense_2/bias/Assign^vc/dense_2/kernel/Adam/Assign ^vc/dense_2/kernel/Adam_1/Assign^vc/dense_2/kernel/Assign^vf/dense/bias/Adam/Assign^vf/dense/bias/Adam_1/Assign^vf/dense/bias/Assign^vf/dense/kernel/Adam/Assign^vf/dense/kernel/Adam_1/Assign^vf/dense/kernel/Assign^vf/dense_1/bias/Adam/Assign^vf/dense_1/bias/Adam_1/Assign^vf/dense_1/bias/Assign^vf/dense_1/kernel/Adam/Assign ^vf/dense_1/kernel/Adam_1/Assign^vf/dense_1/kernel/Assign^vf/dense_2/bias/Adam/Assign^vf/dense_2/bias/Adam_1/Assign^vf/dense_2/bias/Assign^vf/dense_2/kernel/Adam/Assign ^vf/dense_2/kernel/Adam_1/Assign^vf/dense_2/kernel/Assign
c
Reshape_90/shapeConst*
dtype0*
_output_shapes
:*
valueB:
?????????
q

Reshape_90Reshapepi/dense/kernel/readReshape_90/shape*
_output_shapes	
:?x*
Tshape0*
T0
c
Reshape_91/shapeConst*
valueB:
?????????*
_output_shapes
:*
dtype0
o

Reshape_91Reshapepi/dense/bias/readReshape_91/shape*
_output_shapes	
:?*
Tshape0*
T0
c
Reshape_92/shapeConst*
dtype0*
valueB:
?????????*
_output_shapes
:
t

Reshape_92Reshapepi/dense_1/kernel/readReshape_92/shape*
T0*
_output_shapes

:??*
Tshape0
c
Reshape_93/shapeConst*
dtype0*
_output_shapes
:*
valueB:
?????????
q

Reshape_93Reshapepi/dense_1/bias/readReshape_93/shape*
_output_shapes	
:?*
T0*
Tshape0
c
Reshape_94/shapeConst*
valueB:
?????????*
dtype0*
_output_shapes
:
s

Reshape_94Reshapepi/dense_2/kernel/readReshape_94/shape*
_output_shapes	
:?*
T0*
Tshape0
c
Reshape_95/shapeConst*
dtype0*
valueB:
?????????*
_output_shapes
:
p

Reshape_95Reshapepi/dense_2/bias/readReshape_95/shape*
Tshape0*
_output_shapes
:*
T0
c
Reshape_96/shapeConst*
_output_shapes
:*
valueB:
?????????*
dtype0
k

Reshape_96Reshapepi/log_std/readReshape_96/shape*
Tshape0*
_output_shapes
:*
T0
c
Reshape_97/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
q

Reshape_97Reshapevf/dense/kernel/readReshape_97/shape*
T0*
_output_shapes	
:?x*
Tshape0
c
Reshape_98/shapeConst*
dtype0*
_output_shapes
:*
valueB:
?????????
o

Reshape_98Reshapevf/dense/bias/readReshape_98/shape*
_output_shapes	
:?*
Tshape0*
T0
c
Reshape_99/shapeConst*
dtype0*
_output_shapes
:*
valueB:
?????????
t

Reshape_99Reshapevf/dense_1/kernel/readReshape_99/shape*
T0*
_output_shapes

:??*
Tshape0
d
Reshape_100/shapeConst*
dtype0*
_output_shapes
:*
valueB:
?????????
s
Reshape_100Reshapevf/dense_1/bias/readReshape_100/shape*
_output_shapes	
:?*
T0*
Tshape0
d
Reshape_101/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
u
Reshape_101Reshapevf/dense_2/kernel/readReshape_101/shape*
_output_shapes	
:?*
T0*
Tshape0
d
Reshape_102/shapeConst*
_output_shapes
:*
valueB:
?????????*
dtype0
r
Reshape_102Reshapevf/dense_2/bias/readReshape_102/shape*
T0*
Tshape0*
_output_shapes
:
d
Reshape_103/shapeConst*
valueB:
?????????*
dtype0*
_output_shapes
:
s
Reshape_103Reshapevc/dense/kernel/readReshape_103/shape*
T0*
_output_shapes	
:?x*
Tshape0
d
Reshape_104/shapeConst*
valueB:
?????????*
_output_shapes
:*
dtype0
q
Reshape_104Reshapevc/dense/bias/readReshape_104/shape*
T0*
_output_shapes	
:?*
Tshape0
d
Reshape_105/shapeConst*
_output_shapes
:*
valueB:
?????????*
dtype0
v
Reshape_105Reshapevc/dense_1/kernel/readReshape_105/shape*
T0*
Tshape0*
_output_shapes

:??
d
Reshape_106/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
s
Reshape_106Reshapevc/dense_1/bias/readReshape_106/shape*
Tshape0*
_output_shapes	
:?*
T0
d
Reshape_107/shapeConst*
valueB:
?????????*
_output_shapes
:*
dtype0
u
Reshape_107Reshapevc/dense_2/kernel/readReshape_107/shape*
_output_shapes	
:?*
T0*
Tshape0
d
Reshape_108/shapeConst*
valueB:
?????????*
_output_shapes
:*
dtype0
r
Reshape_108Reshapevc/dense_2/bias/readReshape_108/shape*
T0*
_output_shapes
:*
Tshape0
d
Reshape_109/shapeConst*
valueB:
?????????*
_output_shapes
:*
dtype0
n
Reshape_109Reshapebeta1_power/readReshape_109/shape*
Tshape0*
_output_shapes
:*
T0
d
Reshape_110/shapeConst*
dtype0*
_output_shapes
:*
valueB:
?????????
n
Reshape_110Reshapebeta2_power/readReshape_110/shape*
Tshape0*
_output_shapes
:*
T0
d
Reshape_111/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
x
Reshape_111Reshapevf/dense/kernel/Adam/readReshape_111/shape*
_output_shapes	
:?x*
T0*
Tshape0
d
Reshape_112/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
z
Reshape_112Reshapevf/dense/kernel/Adam_1/readReshape_112/shape*
_output_shapes	
:?x*
T0*
Tshape0
d
Reshape_113/shapeConst*
valueB:
?????????*
dtype0*
_output_shapes
:
v
Reshape_113Reshapevf/dense/bias/Adam/readReshape_113/shape*
Tshape0*
_output_shapes	
:?*
T0
d
Reshape_114/shapeConst*
dtype0*
valueB:
?????????*
_output_shapes
:
x
Reshape_114Reshapevf/dense/bias/Adam_1/readReshape_114/shape*
Tshape0*
T0*
_output_shapes	
:?
d
Reshape_115/shapeConst*
dtype0*
valueB:
?????????*
_output_shapes
:
{
Reshape_115Reshapevf/dense_1/kernel/Adam/readReshape_115/shape*
T0*
Tshape0*
_output_shapes

:??
d
Reshape_116/shapeConst*
_output_shapes
:*
valueB:
?????????*
dtype0
}
Reshape_116Reshapevf/dense_1/kernel/Adam_1/readReshape_116/shape*
T0*
Tshape0*
_output_shapes

:??
d
Reshape_117/shapeConst*
_output_shapes
:*
valueB:
?????????*
dtype0
x
Reshape_117Reshapevf/dense_1/bias/Adam/readReshape_117/shape*
Tshape0*
_output_shapes	
:?*
T0
d
Reshape_118/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????
z
Reshape_118Reshapevf/dense_1/bias/Adam_1/readReshape_118/shape*
_output_shapes	
:?*
Tshape0*
T0
d
Reshape_119/shapeConst*
dtype0*
_output_shapes
:*
valueB:
?????????
z
Reshape_119Reshapevf/dense_2/kernel/Adam/readReshape_119/shape*
_output_shapes	
:?*
Tshape0*
T0
d
Reshape_120/shapeConst*
valueB:
?????????*
_output_shapes
:*
dtype0
|
Reshape_120Reshapevf/dense_2/kernel/Adam_1/readReshape_120/shape*
_output_shapes	
:?*
Tshape0*
T0
d
Reshape_121/shapeConst*
dtype0*
valueB:
?????????*
_output_shapes
:
w
Reshape_121Reshapevf/dense_2/bias/Adam/readReshape_121/shape*
_output_shapes
:*
Tshape0*
T0
d
Reshape_122/shapeConst*
_output_shapes
:*
valueB:
?????????*
dtype0
y
Reshape_122Reshapevf/dense_2/bias/Adam_1/readReshape_122/shape*
_output_shapes
:*
Tshape0*
T0
d
Reshape_123/shapeConst*
_output_shapes
:*
valueB:
?????????*
dtype0
x
Reshape_123Reshapevc/dense/kernel/Adam/readReshape_123/shape*
Tshape0*
T0*
_output_shapes	
:?x
d
Reshape_124/shapeConst*
_output_shapes
:*
valueB:
?????????*
dtype0
z
Reshape_124Reshapevc/dense/kernel/Adam_1/readReshape_124/shape*
_output_shapes	
:?x*
T0*
Tshape0
d
Reshape_125/shapeConst*
dtype0*
_output_shapes
:*
valueB:
?????????
v
Reshape_125Reshapevc/dense/bias/Adam/readReshape_125/shape*
Tshape0*
T0*
_output_shapes	
:?
d
Reshape_126/shapeConst*
valueB:
?????????*
_output_shapes
:*
dtype0
x
Reshape_126Reshapevc/dense/bias/Adam_1/readReshape_126/shape*
Tshape0*
T0*
_output_shapes	
:?
d
Reshape_127/shapeConst*
dtype0*
valueB:
?????????*
_output_shapes
:
{
Reshape_127Reshapevc/dense_1/kernel/Adam/readReshape_127/shape*
Tshape0*
T0*
_output_shapes

:??
d
Reshape_128/shapeConst*
valueB:
?????????*
_output_shapes
:*
dtype0
}
Reshape_128Reshapevc/dense_1/kernel/Adam_1/readReshape_128/shape*
T0*
Tshape0*
_output_shapes

:??
d
Reshape_129/shapeConst*
_output_shapes
:*
valueB:
?????????*
dtype0
x
Reshape_129Reshapevc/dense_1/bias/Adam/readReshape_129/shape*
_output_shapes	
:?*
T0*
Tshape0
d
Reshape_130/shapeConst*
dtype0*
valueB:
?????????*
_output_shapes
:
z
Reshape_130Reshapevc/dense_1/bias/Adam_1/readReshape_130/shape*
Tshape0*
T0*
_output_shapes	
:?
d
Reshape_131/shapeConst*
dtype0*
valueB:
?????????*
_output_shapes
:
z
Reshape_131Reshapevc/dense_2/kernel/Adam/readReshape_131/shape*
T0*
Tshape0*
_output_shapes	
:?
d
Reshape_132/shapeConst*
_output_shapes
:*
valueB:
?????????*
dtype0
|
Reshape_132Reshapevc/dense_2/kernel/Adam_1/readReshape_132/shape*
T0*
_output_shapes	
:?*
Tshape0
d
Reshape_133/shapeConst*
dtype0*
valueB:
?????????*
_output_shapes
:
w
Reshape_133Reshapevc/dense_2/bias/Adam/readReshape_133/shape*
Tshape0*
_output_shapes
:*
T0
d
Reshape_134/shapeConst*
dtype0*
_output_shapes
:*
valueB:
?????????
y
Reshape_134Reshapevc/dense_2/bias/Adam_1/readReshape_134/shape*
_output_shapes
:*
T0*
Tshape0
O
concat_7/axisConst*
_output_shapes
: *
value	B : *
dtype0
?
concat_7ConcatV2
Reshape_90
Reshape_91
Reshape_92
Reshape_93
Reshape_94
Reshape_95
Reshape_96
Reshape_97
Reshape_98
Reshape_99Reshape_100Reshape_101Reshape_102Reshape_103Reshape_104Reshape_105Reshape_106Reshape_107Reshape_108Reshape_109Reshape_110Reshape_111Reshape_112Reshape_113Reshape_114Reshape_115Reshape_116Reshape_117Reshape_118Reshape_119Reshape_120Reshape_121Reshape_122Reshape_123Reshape_124Reshape_125Reshape_126Reshape_127Reshape_128Reshape_129Reshape_130Reshape_131Reshape_132Reshape_133Reshape_134concat_7/axis*
_output_shapes

:??"*

Tidx0*
N-*
T0
h
PyFunc_2PyFuncconcat_7*
_output_shapes
:*
Tin
2*
token
pyfunc_2*
Tout
2
?
Const_9Const*
dtype0*?
value?B?-"? <                     <                  <                        <   <                                 <   <                                *
_output_shapes
:-
S
split_3/split_dimConst*
_output_shapes
: *
dtype0*
value	B : 
?
split_3SplitVPyFunc_2Const_9split_3/split_dim*

Tlen0*
	num_split-*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::::*
T0
b
Reshape_135/shapeConst*
valueB"<      *
_output_shapes
:*
dtype0
j
Reshape_135Reshapesplit_3Reshape_135/shape*
_output_shapes
:	<?*
T0*
Tshape0
\
Reshape_136/shapeConst*
valueB:?*
dtype0*
_output_shapes
:
h
Reshape_136Reshape	split_3:1Reshape_136/shape*
Tshape0*
T0*
_output_shapes	
:?
b
Reshape_137/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
m
Reshape_137Reshape	split_3:2Reshape_137/shape*
Tshape0* 
_output_shapes
:
??*
T0
\
Reshape_138/shapeConst*
dtype0*
_output_shapes
:*
valueB:?
h
Reshape_138Reshape	split_3:3Reshape_138/shape*
_output_shapes	
:?*
Tshape0*
T0
b
Reshape_139/shapeConst*
_output_shapes
:*
dtype0*
valueB"      
l
Reshape_139Reshape	split_3:4Reshape_139/shape*
Tshape0*
T0*
_output_shapes
:	?
[
Reshape_140/shapeConst*
valueB:*
_output_shapes
:*
dtype0
g
Reshape_140Reshape	split_3:5Reshape_140/shape*
_output_shapes
:*
Tshape0*
T0
[
Reshape_141/shapeConst*
_output_shapes
:*
dtype0*
valueB:
g
Reshape_141Reshape	split_3:6Reshape_141/shape*
Tshape0*
T0*
_output_shapes
:
b
Reshape_142/shapeConst*
valueB"<      *
_output_shapes
:*
dtype0
l
Reshape_142Reshape	split_3:7Reshape_142/shape*
T0*
_output_shapes
:	<?*
Tshape0
\
Reshape_143/shapeConst*
valueB:?*
dtype0*
_output_shapes
:
h
Reshape_143Reshape	split_3:8Reshape_143/shape*
Tshape0*
T0*
_output_shapes	
:?
b
Reshape_144/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
m
Reshape_144Reshape	split_3:9Reshape_144/shape*
Tshape0*
T0* 
_output_shapes
:
??
\
Reshape_145/shapeConst*
_output_shapes
:*
valueB:?*
dtype0
i
Reshape_145Reshape
split_3:10Reshape_145/shape*
T0*
_output_shapes	
:?*
Tshape0
b
Reshape_146/shapeConst*
_output_shapes
:*
valueB"      *
dtype0
m
Reshape_146Reshape
split_3:11Reshape_146/shape*
_output_shapes
:	?*
T0*
Tshape0
[
Reshape_147/shapeConst*
dtype0*
_output_shapes
:*
valueB:
h
Reshape_147Reshape
split_3:12Reshape_147/shape*
Tshape0*
_output_shapes
:*
T0
b
Reshape_148/shapeConst*
dtype0*
valueB"<      *
_output_shapes
:
m
Reshape_148Reshape
split_3:13Reshape_148/shape*
_output_shapes
:	<?*
Tshape0*
T0
\
Reshape_149/shapeConst*
_output_shapes
:*
dtype0*
valueB:?
i
Reshape_149Reshape
split_3:14Reshape_149/shape*
_output_shapes	
:?*
T0*
Tshape0
b
Reshape_150/shapeConst*
dtype0*
_output_shapes
:*
valueB"      
n
Reshape_150Reshape
split_3:15Reshape_150/shape*
Tshape0* 
_output_shapes
:
??*
T0
\
Reshape_151/shapeConst*
valueB:?*
_output_shapes
:*
dtype0
i
Reshape_151Reshape
split_3:16Reshape_151/shape*
_output_shapes	
:?*
T0*
Tshape0
b
Reshape_152/shapeConst*
_output_shapes
:*
dtype0*
valueB"      
m
Reshape_152Reshape
split_3:17Reshape_152/shape*
_output_shapes
:	?*
Tshape0*
T0
[
Reshape_153/shapeConst*
_output_shapes
:*
valueB:*
dtype0
h
Reshape_153Reshape
split_3:18Reshape_153/shape*
_output_shapes
:*
T0*
Tshape0
T
Reshape_154/shapeConst*
valueB *
dtype0*
_output_shapes
: 
d
Reshape_154Reshape
split_3:19Reshape_154/shape*
Tshape0*
T0*
_output_shapes
: 
T
Reshape_155/shapeConst*
dtype0*
_output_shapes
: *
valueB 
d
Reshape_155Reshape
split_3:20Reshape_155/shape*
T0*
Tshape0*
_output_shapes
: 
b
Reshape_156/shapeConst*
valueB"<      *
_output_shapes
:*
dtype0
m
Reshape_156Reshape
split_3:21Reshape_156/shape*
T0*
_output_shapes
:	<?*
Tshape0
b
Reshape_157/shapeConst*
dtype0*
_output_shapes
:*
valueB"<      
m
Reshape_157Reshape
split_3:22Reshape_157/shape*
T0*
_output_shapes
:	<?*
Tshape0
\
Reshape_158/shapeConst*
valueB:?*
dtype0*
_output_shapes
:
i
Reshape_158Reshape
split_3:23Reshape_158/shape*
_output_shapes	
:?*
Tshape0*
T0
\
Reshape_159/shapeConst*
valueB:?*
_output_shapes
:*
dtype0
i
Reshape_159Reshape
split_3:24Reshape_159/shape*
T0*
_output_shapes	
:?*
Tshape0
b
Reshape_160/shapeConst*
valueB"      *
_output_shapes
:*
dtype0
n
Reshape_160Reshape
split_3:25Reshape_160/shape*
Tshape0*
T0* 
_output_shapes
:
??
b
Reshape_161/shapeConst*
valueB"      *
_output_shapes
:*
dtype0
n
Reshape_161Reshape
split_3:26Reshape_161/shape*
T0* 
_output_shapes
:
??*
Tshape0
\
Reshape_162/shapeConst*
_output_shapes
:*
dtype0*
valueB:?
i
Reshape_162Reshape
split_3:27Reshape_162/shape*
Tshape0*
_output_shapes	
:?*
T0
\
Reshape_163/shapeConst*
_output_shapes
:*
dtype0*
valueB:?
i
Reshape_163Reshape
split_3:28Reshape_163/shape*
_output_shapes	
:?*
T0*
Tshape0
b
Reshape_164/shapeConst*
_output_shapes
:*
valueB"      *
dtype0
m
Reshape_164Reshape
split_3:29Reshape_164/shape*
_output_shapes
:	?*
T0*
Tshape0
b
Reshape_165/shapeConst*
dtype0*
valueB"      *
_output_shapes
:
m
Reshape_165Reshape
split_3:30Reshape_165/shape*
Tshape0*
T0*
_output_shapes
:	?
[
Reshape_166/shapeConst*
valueB:*
_output_shapes
:*
dtype0
h
Reshape_166Reshape
split_3:31Reshape_166/shape*
T0*
_output_shapes
:*
Tshape0
[
Reshape_167/shapeConst*
_output_shapes
:*
dtype0*
valueB:
h
Reshape_167Reshape
split_3:32Reshape_167/shape*
T0*
_output_shapes
:*
Tshape0
b
Reshape_168/shapeConst*
dtype0*
_output_shapes
:*
valueB"<      
m
Reshape_168Reshape
split_3:33Reshape_168/shape*
Tshape0*
T0*
_output_shapes
:	<?
b
Reshape_169/shapeConst*
_output_shapes
:*
dtype0*
valueB"<      
m
Reshape_169Reshape
split_3:34Reshape_169/shape*
_output_shapes
:	<?*
T0*
Tshape0
\
Reshape_170/shapeConst*
valueB:?*
_output_shapes
:*
dtype0
i
Reshape_170Reshape
split_3:35Reshape_170/shape*
_output_shapes	
:?*
Tshape0*
T0
\
Reshape_171/shapeConst*
dtype0*
_output_shapes
:*
valueB:?
i
Reshape_171Reshape
split_3:36Reshape_171/shape*
_output_shapes	
:?*
T0*
Tshape0
b
Reshape_172/shapeConst*
dtype0*
valueB"      *
_output_shapes
:
n
Reshape_172Reshape
split_3:37Reshape_172/shape*
Tshape0*
T0* 
_output_shapes
:
??
b
Reshape_173/shapeConst*
valueB"      *
dtype0*
_output_shapes
:
n
Reshape_173Reshape
split_3:38Reshape_173/shape*
T0* 
_output_shapes
:
??*
Tshape0
\
Reshape_174/shapeConst*
_output_shapes
:*
dtype0*
valueB:?
i
Reshape_174Reshape
split_3:39Reshape_174/shape*
_output_shapes	
:?*
Tshape0*
T0
\
Reshape_175/shapeConst*
_output_shapes
:*
valueB:?*
dtype0
i
Reshape_175Reshape
split_3:40Reshape_175/shape*
Tshape0*
_output_shapes	
:?*
T0
b
Reshape_176/shapeConst*
_output_shapes
:*
valueB"      *
dtype0
m
Reshape_176Reshape
split_3:41Reshape_176/shape*
Tshape0*
T0*
_output_shapes
:	?
b
Reshape_177/shapeConst*
dtype0*
valueB"      *
_output_shapes
:
m
Reshape_177Reshape
split_3:42Reshape_177/shape*
Tshape0*
T0*
_output_shapes
:	?
[
Reshape_178/shapeConst*
_output_shapes
:*
valueB:*
dtype0
h
Reshape_178Reshape
split_3:43Reshape_178/shape*
Tshape0*
T0*
_output_shapes
:
[
Reshape_179/shapeConst*
_output_shapes
:*
valueB:*
dtype0
h
Reshape_179Reshape
split_3:44Reshape_179/shape*
_output_shapes
:*
Tshape0*
T0
?
	Assign_19Assignpi/dense/kernelReshape_135*"
_class
loc:@pi/dense/kernel*
_output_shapes
:	<?*
validate_shape(*
T0*
use_locking(
?
	Assign_20Assignpi/dense/biasReshape_136*
validate_shape(*
T0*
use_locking(* 
_class
loc:@pi/dense/bias*
_output_shapes	
:?
?
	Assign_21Assignpi/dense_1/kernelReshape_137*
use_locking(* 
_output_shapes
:
??*
validate_shape(*
T0*$
_class
loc:@pi/dense_1/kernel
?
	Assign_22Assignpi/dense_1/biasReshape_138*"
_class
loc:@pi/dense_1/bias*
T0*
use_locking(*
_output_shapes	
:?*
validate_shape(
?
	Assign_23Assignpi/dense_2/kernelReshape_139*$
_class
loc:@pi/dense_2/kernel*
T0*
validate_shape(*
use_locking(*
_output_shapes
:	?
?
	Assign_24Assignpi/dense_2/biasReshape_140*
use_locking(*
_output_shapes
:*"
_class
loc:@pi/dense_2/bias*
T0*
validate_shape(
?
	Assign_25Assign
pi/log_stdReshape_141*
T0*
use_locking(*
_output_shapes
:*
_class
loc:@pi/log_std*
validate_shape(
?
	Assign_26Assignvf/dense/kernelReshape_142*
validate_shape(*
use_locking(*
_output_shapes
:	<?*
T0*"
_class
loc:@vf/dense/kernel
?
	Assign_27Assignvf/dense/biasReshape_143* 
_class
loc:@vf/dense/bias*
validate_shape(*
T0*
use_locking(*
_output_shapes	
:?
?
	Assign_28Assignvf/dense_1/kernelReshape_144*
use_locking(*
validate_shape(*
T0* 
_output_shapes
:
??*$
_class
loc:@vf/dense_1/kernel
?
	Assign_29Assignvf/dense_1/biasReshape_145*
T0*"
_class
loc:@vf/dense_1/bias*
_output_shapes	
:?*
use_locking(*
validate_shape(
?
	Assign_30Assignvf/dense_2/kernelReshape_146*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	?*
T0*
use_locking(*
validate_shape(
?
	Assign_31Assignvf/dense_2/biasReshape_147*
validate_shape(*
T0*
use_locking(*"
_class
loc:@vf/dense_2/bias*
_output_shapes
:
?
	Assign_32Assignvc/dense/kernelReshape_148*
validate_shape(*
use_locking(*"
_class
loc:@vc/dense/kernel*
T0*
_output_shapes
:	<?
?
	Assign_33Assignvc/dense/biasReshape_149*
_output_shapes	
:?*
validate_shape(* 
_class
loc:@vc/dense/bias*
T0*
use_locking(
?
	Assign_34Assignvc/dense_1/kernelReshape_150*$
_class
loc:@vc/dense_1/kernel*
use_locking(* 
_output_shapes
:
??*
T0*
validate_shape(
?
	Assign_35Assignvc/dense_1/biasReshape_151*
use_locking(*
validate_shape(*
T0*"
_class
loc:@vc/dense_1/bias*
_output_shapes	
:?
?
	Assign_36Assignvc/dense_2/kernelReshape_152*
T0*
_output_shapes
:	?*
validate_shape(*$
_class
loc:@vc/dense_2/kernel*
use_locking(
?
	Assign_37Assignvc/dense_2/biasReshape_153*
validate_shape(*
use_locking(*
_output_shapes
:*"
_class
loc:@vc/dense_2/bias*
T0
?
	Assign_38Assignbeta1_powerReshape_154* 
_class
loc:@vc/dense/bias*
validate_shape(*
T0*
use_locking(*
_output_shapes
: 
?
	Assign_39Assignbeta2_powerReshape_155*
_output_shapes
: * 
_class
loc:@vc/dense/bias*
validate_shape(*
T0*
use_locking(
?
	Assign_40Assignvf/dense/kernel/AdamReshape_156*
_output_shapes
:	<?*
validate_shape(*
use_locking(*
T0*"
_class
loc:@vf/dense/kernel
?
	Assign_41Assignvf/dense/kernel/Adam_1Reshape_157*
validate_shape(*
T0*"
_class
loc:@vf/dense/kernel*
use_locking(*
_output_shapes
:	<?
?
	Assign_42Assignvf/dense/bias/AdamReshape_158*
use_locking(* 
_class
loc:@vf/dense/bias*
T0*
validate_shape(*
_output_shapes	
:?
?
	Assign_43Assignvf/dense/bias/Adam_1Reshape_159*
validate_shape(* 
_class
loc:@vf/dense/bias*
_output_shapes	
:?*
use_locking(*
T0
?
	Assign_44Assignvf/dense_1/kernel/AdamReshape_160*
use_locking(*
T0*
validate_shape(*$
_class
loc:@vf/dense_1/kernel* 
_output_shapes
:
??
?
	Assign_45Assignvf/dense_1/kernel/Adam_1Reshape_161*
T0*
validate_shape(*
use_locking(*$
_class
loc:@vf/dense_1/kernel* 
_output_shapes
:
??
?
	Assign_46Assignvf/dense_1/bias/AdamReshape_162*
_output_shapes	
:?*
use_locking(*
T0*
validate_shape(*"
_class
loc:@vf/dense_1/bias
?
	Assign_47Assignvf/dense_1/bias/Adam_1Reshape_163*
use_locking(*"
_class
loc:@vf/dense_1/bias*
_output_shapes	
:?*
T0*
validate_shape(
?
	Assign_48Assignvf/dense_2/kernel/AdamReshape_164*
use_locking(*
_output_shapes
:	?*
T0*$
_class
loc:@vf/dense_2/kernel*
validate_shape(
?
	Assign_49Assignvf/dense_2/kernel/Adam_1Reshape_165*
_output_shapes
:	?*
use_locking(*$
_class
loc:@vf/dense_2/kernel*
T0*
validate_shape(
?
	Assign_50Assignvf/dense_2/bias/AdamReshape_166*"
_class
loc:@vf/dense_2/bias*
_output_shapes
:*
use_locking(*
validate_shape(*
T0
?
	Assign_51Assignvf/dense_2/bias/Adam_1Reshape_167*
_output_shapes
:*
validate_shape(*
T0*"
_class
loc:@vf/dense_2/bias*
use_locking(
?
	Assign_52Assignvc/dense/kernel/AdamReshape_168*
validate_shape(*
T0*
use_locking(*
_output_shapes
:	<?*"
_class
loc:@vc/dense/kernel
?
	Assign_53Assignvc/dense/kernel/Adam_1Reshape_169*
use_locking(*
_output_shapes
:	<?*"
_class
loc:@vc/dense/kernel*
T0*
validate_shape(
?
	Assign_54Assignvc/dense/bias/AdamReshape_170* 
_class
loc:@vc/dense/bias*
_output_shapes	
:?*
T0*
validate_shape(*
use_locking(
?
	Assign_55Assignvc/dense/bias/Adam_1Reshape_171* 
_class
loc:@vc/dense/bias*
use_locking(*
_output_shapes	
:?*
T0*
validate_shape(
?
	Assign_56Assignvc/dense_1/kernel/AdamReshape_172*
T0* 
_output_shapes
:
??*
validate_shape(*$
_class
loc:@vc/dense_1/kernel*
use_locking(
?
	Assign_57Assignvc/dense_1/kernel/Adam_1Reshape_173*
T0* 
_output_shapes
:
??*
use_locking(*$
_class
loc:@vc/dense_1/kernel*
validate_shape(
?
	Assign_58Assignvc/dense_1/bias/AdamReshape_174*
use_locking(*"
_class
loc:@vc/dense_1/bias*
T0*
_output_shapes	
:?*
validate_shape(
?
	Assign_59Assignvc/dense_1/bias/Adam_1Reshape_175*"
_class
loc:@vc/dense_1/bias*
use_locking(*
_output_shapes	
:?*
validate_shape(*
T0
?
	Assign_60Assignvc/dense_2/kernel/AdamReshape_176*
T0*
validate_shape(*$
_class
loc:@vc/dense_2/kernel*
use_locking(*
_output_shapes
:	?
?
	Assign_61Assignvc/dense_2/kernel/Adam_1Reshape_177*$
_class
loc:@vc/dense_2/kernel*
use_locking(*
T0*
validate_shape(*
_output_shapes
:	?
?
	Assign_62Assignvc/dense_2/bias/AdamReshape_178*
T0*
use_locking(*
validate_shape(*"
_class
loc:@vc/dense_2/bias*
_output_shapes
:
?
	Assign_63Assignvc/dense_2/bias/Adam_1Reshape_179*
T0*"
_class
loc:@vc/dense_2/bias*
_output_shapes
:*
validate_shape(*
use_locking(
?
group_deps_3NoOp
^Assign_19
^Assign_20
^Assign_21
^Assign_22
^Assign_23
^Assign_24
^Assign_25
^Assign_26
^Assign_27
^Assign_28
^Assign_29
^Assign_30
^Assign_31
^Assign_32
^Assign_33
^Assign_34
^Assign_35
^Assign_36
^Assign_37
^Assign_38
^Assign_39
^Assign_40
^Assign_41
^Assign_42
^Assign_43
^Assign_44
^Assign_45
^Assign_46
^Assign_47
^Assign_48
^Assign_49
^Assign_50
^Assign_51
^Assign_52
^Assign_53
^Assign_54
^Assign_55
^Assign_56
^Assign_57
^Assign_58
^Assign_59
^Assign_60
^Assign_61
^Assign_62
^Assign_63
Y
save/filename/inputConst*
_output_shapes
: *
valueB Bmodel*
dtype0
n
save/filenamePlaceholderWithDefaultsave/filename/input*
_output_shapes
: *
dtype0*
shape: 
e

save/ConstPlaceholderWithDefaultsave/filename*
dtype0*
shape: *
_output_shapes
: 
?
save/StringJoin/inputs_1Const*<
value3B1 B+_temp_efb7a7d151a24736ba68fe5848f50890/part*
dtype0*
_output_shapes
: 
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
_output_shapes
: *
N*
	separator 
Q
save/num_shardsConst*
dtype0*
_output_shapes
: *
value	B :
\
save/ShardedFilename/shardConst*
_output_shapes
: *
value	B : *
dtype0
}
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards*
_output_shapes
: 
?
save/SaveV2/tensor_namesConst*
_output_shapes
:-*
dtype0*?
value?B?-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1
?
save/SaveV2/shape_and_slicesConst*
_output_shapes
:-*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
?
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesbeta1_powerbeta2_powerpi/dense/biaspi/dense/kernelpi/dense_1/biaspi/dense_1/kernelpi/dense_2/biaspi/dense_2/kernel
pi/log_stdvc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*;
dtypes1
/2-
?
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2*
_output_shapes
: *'
_class
loc:@save/ShardedFilename*
T0
?
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency*
T0*
N*

axis *
_output_shapes
:
}
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const*
delete_old_dirs(
z
save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency*
_output_shapes
: *
T0
?
save/RestoreV2/tensor_namesConst*
_output_shapes
:-*
dtype0*?
value?B?-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1
?
save/RestoreV2/shape_and_slicesConst*
dtype0*
_output_shapes
:-*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 
?
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::::*;
dtypes1
/2-
?
save/AssignAssignbeta1_powersave/RestoreV2*
validate_shape(*
use_locking(* 
_class
loc:@vc/dense/bias*
_output_shapes
: *
T0
?
save/Assign_1Assignbeta2_powersave/RestoreV2:1* 
_class
loc:@vc/dense/bias*
validate_shape(*
use_locking(*
T0*
_output_shapes
: 
?
save/Assign_2Assignpi/dense/biassave/RestoreV2:2*
use_locking(*
validate_shape(*
T0* 
_class
loc:@pi/dense/bias*
_output_shapes	
:?
?
save/Assign_3Assignpi/dense/kernelsave/RestoreV2:3*
validate_shape(*"
_class
loc:@pi/dense/kernel*
T0*
_output_shapes
:	<?*
use_locking(
?
save/Assign_4Assignpi/dense_1/biassave/RestoreV2:4*
use_locking(*"
_class
loc:@pi/dense_1/bias*
T0*
validate_shape(*
_output_shapes	
:?
?
save/Assign_5Assignpi/dense_1/kernelsave/RestoreV2:5*
T0*
use_locking(*
validate_shape(*$
_class
loc:@pi/dense_1/kernel* 
_output_shapes
:
??
?
save/Assign_6Assignpi/dense_2/biassave/RestoreV2:6*
T0*
validate_shape(*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:*
use_locking(
?
save/Assign_7Assignpi/dense_2/kernelsave/RestoreV2:7*
_output_shapes
:	?*
use_locking(*
T0*$
_class
loc:@pi/dense_2/kernel*
validate_shape(
?
save/Assign_8Assign
pi/log_stdsave/RestoreV2:8*
use_locking(*
_class
loc:@pi/log_std*
T0*
validate_shape(*
_output_shapes
:
?
save/Assign_9Assignvc/dense/biassave/RestoreV2:9*
_output_shapes	
:?* 
_class
loc:@vc/dense/bias*
use_locking(*
T0*
validate_shape(
?
save/Assign_10Assignvc/dense/bias/Adamsave/RestoreV2:10*
_output_shapes	
:?* 
_class
loc:@vc/dense/bias*
use_locking(*
T0*
validate_shape(
?
save/Assign_11Assignvc/dense/bias/Adam_1save/RestoreV2:11*
validate_shape(*
T0*
use_locking(*
_output_shapes	
:?* 
_class
loc:@vc/dense/bias
?
save/Assign_12Assignvc/dense/kernelsave/RestoreV2:12*
validate_shape(*
T0*"
_class
loc:@vc/dense/kernel*
_output_shapes
:	<?*
use_locking(
?
save/Assign_13Assignvc/dense/kernel/Adamsave/RestoreV2:13*
T0*
_output_shapes
:	<?*"
_class
loc:@vc/dense/kernel*
use_locking(*
validate_shape(
?
save/Assign_14Assignvc/dense/kernel/Adam_1save/RestoreV2:14*"
_class
loc:@vc/dense/kernel*
validate_shape(*
use_locking(*
_output_shapes
:	<?*
T0
?
save/Assign_15Assignvc/dense_1/biassave/RestoreV2:15*
T0*
use_locking(*"
_class
loc:@vc/dense_1/bias*
_output_shapes	
:?*
validate_shape(
?
save/Assign_16Assignvc/dense_1/bias/Adamsave/RestoreV2:16*
T0*
use_locking(*
validate_shape(*
_output_shapes	
:?*"
_class
loc:@vc/dense_1/bias
?
save/Assign_17Assignvc/dense_1/bias/Adam_1save/RestoreV2:17*
_output_shapes	
:?*
validate_shape(*
use_locking(*
T0*"
_class
loc:@vc/dense_1/bias
?
save/Assign_18Assignvc/dense_1/kernelsave/RestoreV2:18* 
_output_shapes
:
??*
T0*
use_locking(*
validate_shape(*$
_class
loc:@vc/dense_1/kernel
?
save/Assign_19Assignvc/dense_1/kernel/Adamsave/RestoreV2:19*
T0* 
_output_shapes
:
??*
validate_shape(*$
_class
loc:@vc/dense_1/kernel*
use_locking(
?
save/Assign_20Assignvc/dense_1/kernel/Adam_1save/RestoreV2:20* 
_output_shapes
:
??*
use_locking(*$
_class
loc:@vc/dense_1/kernel*
validate_shape(*
T0
?
save/Assign_21Assignvc/dense_2/biassave/RestoreV2:21*"
_class
loc:@vc/dense_2/bias*
T0*
validate_shape(*
_output_shapes
:*
use_locking(
?
save/Assign_22Assignvc/dense_2/bias/Adamsave/RestoreV2:22*
use_locking(*
_output_shapes
:*
validate_shape(*"
_class
loc:@vc/dense_2/bias*
T0
?
save/Assign_23Assignvc/dense_2/bias/Adam_1save/RestoreV2:23*
use_locking(*"
_class
loc:@vc/dense_2/bias*
validate_shape(*
T0*
_output_shapes
:
?
save/Assign_24Assignvc/dense_2/kernelsave/RestoreV2:24*
T0*
_output_shapes
:	?*
validate_shape(*
use_locking(*$
_class
loc:@vc/dense_2/kernel
?
save/Assign_25Assignvc/dense_2/kernel/Adamsave/RestoreV2:25*
T0*
_output_shapes
:	?*$
_class
loc:@vc/dense_2/kernel*
use_locking(*
validate_shape(
?
save/Assign_26Assignvc/dense_2/kernel/Adam_1save/RestoreV2:26*
_output_shapes
:	?*
T0*$
_class
loc:@vc/dense_2/kernel*
validate_shape(*
use_locking(
?
save/Assign_27Assignvf/dense/biassave/RestoreV2:27*
T0*
use_locking(* 
_class
loc:@vf/dense/bias*
_output_shapes	
:?*
validate_shape(
?
save/Assign_28Assignvf/dense/bias/Adamsave/RestoreV2:28*
validate_shape(* 
_class
loc:@vf/dense/bias*
T0*
use_locking(*
_output_shapes	
:?
?
save/Assign_29Assignvf/dense/bias/Adam_1save/RestoreV2:29*
_output_shapes	
:?*
use_locking(* 
_class
loc:@vf/dense/bias*
validate_shape(*
T0
?
save/Assign_30Assignvf/dense/kernelsave/RestoreV2:30*"
_class
loc:@vf/dense/kernel*
T0*
validate_shape(*
_output_shapes
:	<?*
use_locking(
?
save/Assign_31Assignvf/dense/kernel/Adamsave/RestoreV2:31*
validate_shape(*"
_class
loc:@vf/dense/kernel*
_output_shapes
:	<?*
use_locking(*
T0
?
save/Assign_32Assignvf/dense/kernel/Adam_1save/RestoreV2:32*
_output_shapes
:	<?*
validate_shape(*
use_locking(*"
_class
loc:@vf/dense/kernel*
T0
?
save/Assign_33Assignvf/dense_1/biassave/RestoreV2:33*
_output_shapes	
:?*
validate_shape(*"
_class
loc:@vf/dense_1/bias*
T0*
use_locking(
?
save/Assign_34Assignvf/dense_1/bias/Adamsave/RestoreV2:34*
T0*
_output_shapes	
:?*
validate_shape(*
use_locking(*"
_class
loc:@vf/dense_1/bias
?
save/Assign_35Assignvf/dense_1/bias/Adam_1save/RestoreV2:35*
T0*
_output_shapes	
:?*
use_locking(*
validate_shape(*"
_class
loc:@vf/dense_1/bias
?
save/Assign_36Assignvf/dense_1/kernelsave/RestoreV2:36*
validate_shape(* 
_output_shapes
:
??*$
_class
loc:@vf/dense_1/kernel*
use_locking(*
T0
?
save/Assign_37Assignvf/dense_1/kernel/Adamsave/RestoreV2:37* 
_output_shapes
:
??*$
_class
loc:@vf/dense_1/kernel*
T0*
validate_shape(*
use_locking(
?
save/Assign_38Assignvf/dense_1/kernel/Adam_1save/RestoreV2:38*
T0*
use_locking(* 
_output_shapes
:
??*
validate_shape(*$
_class
loc:@vf/dense_1/kernel
?
save/Assign_39Assignvf/dense_2/biassave/RestoreV2:39*
T0*
use_locking(*
validate_shape(*
_output_shapes
:*"
_class
loc:@vf/dense_2/bias
?
save/Assign_40Assignvf/dense_2/bias/Adamsave/RestoreV2:40*
T0*"
_class
loc:@vf/dense_2/bias*
use_locking(*
validate_shape(*
_output_shapes
:
?
save/Assign_41Assignvf/dense_2/bias/Adam_1save/RestoreV2:41*
T0*
use_locking(*"
_class
loc:@vf/dense_2/bias*
_output_shapes
:*
validate_shape(
?
save/Assign_42Assignvf/dense_2/kernelsave/RestoreV2:42*
T0*
validate_shape(*$
_class
loc:@vf/dense_2/kernel*
use_locking(*
_output_shapes
:	?
?
save/Assign_43Assignvf/dense_2/kernel/Adamsave/RestoreV2:43*
T0*
validate_shape(*
use_locking(*
_output_shapes
:	?*$
_class
loc:@vf/dense_2/kernel
?
save/Assign_44Assignvf/dense_2/kernel/Adam_1save/RestoreV2:44*
use_locking(*$
_class
loc:@vf/dense_2/kernel*
validate_shape(*
T0*
_output_shapes
:	?
?
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_3^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33^save/Assign_34^save/Assign_35^save/Assign_36^save/Assign_37^save/Assign_38^save/Assign_39^save/Assign_4^save/Assign_40^save/Assign_41^save/Assign_42^save/Assign_43^save/Assign_44^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9
-
save/restore_allNoOp^save/restore_shard
[
save_1/filename/inputConst*
valueB Bmodel*
_output_shapes
: *
dtype0
r
save_1/filenamePlaceholderWithDefaultsave_1/filename/input*
dtype0*
shape: *
_output_shapes
: 
i
save_1/ConstPlaceholderWithDefaultsave_1/filename*
_output_shapes
: *
shape: *
dtype0
?
save_1/StringJoin/inputs_1Const*<
value3B1 B+_temp_5b05565940744740812332d603323c4a/part*
dtype0*
_output_shapes
: 
{
save_1/StringJoin
StringJoinsave_1/Constsave_1/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
S
save_1/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_1/ShardedFilename/shardConst*
_output_shapes
: *
dtype0*
value	B : 
?
save_1/ShardedFilenameShardedFilenamesave_1/StringJoinsave_1/ShardedFilename/shardsave_1/num_shards*
_output_shapes
: 
?
save_1/SaveV2/tensor_namesConst*?
value?B?-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
_output_shapes
:-*
dtype0
?
save_1/SaveV2/shape_and_slicesConst*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:-*
dtype0
?
save_1/SaveV2SaveV2save_1/ShardedFilenamesave_1/SaveV2/tensor_namessave_1/SaveV2/shape_and_slicesbeta1_powerbeta2_powerpi/dense/biaspi/dense/kernelpi/dense_1/biaspi/dense_1/kernelpi/dense_2/biaspi/dense_2/kernel
pi/log_stdvc/dense/biasvc/dense/bias/Adamvc/dense/bias/Adam_1vc/dense/kernelvc/dense/kernel/Adamvc/dense/kernel/Adam_1vc/dense_1/biasvc/dense_1/bias/Adamvc/dense_1/bias/Adam_1vc/dense_1/kernelvc/dense_1/kernel/Adamvc/dense_1/kernel/Adam_1vc/dense_2/biasvc/dense_2/bias/Adamvc/dense_2/bias/Adam_1vc/dense_2/kernelvc/dense_2/kernel/Adamvc/dense_2/kernel/Adam_1vf/dense/biasvf/dense/bias/Adamvf/dense/bias/Adam_1vf/dense/kernelvf/dense/kernel/Adamvf/dense/kernel/Adam_1vf/dense_1/biasvf/dense_1/bias/Adamvf/dense_1/bias/Adam_1vf/dense_1/kernelvf/dense_1/kernel/Adamvf/dense_1/kernel/Adam_1vf/dense_2/biasvf/dense_2/bias/Adamvf/dense_2/bias/Adam_1vf/dense_2/kernelvf/dense_2/kernel/Adamvf/dense_2/kernel/Adam_1*;
dtypes1
/2-
?
save_1/control_dependencyIdentitysave_1/ShardedFilename^save_1/SaveV2*
T0*
_output_shapes
: *)
_class
loc:@save_1/ShardedFilename
?
-save_1/MergeV2Checkpoints/checkpoint_prefixesPacksave_1/ShardedFilename^save_1/control_dependency*

axis *
_output_shapes
:*
T0*
N
?
save_1/MergeV2CheckpointsMergeV2Checkpoints-save_1/MergeV2Checkpoints/checkpoint_prefixessave_1/Const*
delete_old_dirs(
?
save_1/IdentityIdentitysave_1/Const^save_1/MergeV2Checkpoints^save_1/control_dependency*
T0*
_output_shapes
: 
?
save_1/RestoreV2/tensor_namesConst*?
value?B?-Bbeta1_powerBbeta2_powerBpi/dense/biasBpi/dense/kernelBpi/dense_1/biasBpi/dense_1/kernelBpi/dense_2/biasBpi/dense_2/kernelB
pi/log_stdBvc/dense/biasBvc/dense/bias/AdamBvc/dense/bias/Adam_1Bvc/dense/kernelBvc/dense/kernel/AdamBvc/dense/kernel/Adam_1Bvc/dense_1/biasBvc/dense_1/bias/AdamBvc/dense_1/bias/Adam_1Bvc/dense_1/kernelBvc/dense_1/kernel/AdamBvc/dense_1/kernel/Adam_1Bvc/dense_2/biasBvc/dense_2/bias/AdamBvc/dense_2/bias/Adam_1Bvc/dense_2/kernelBvc/dense_2/kernel/AdamBvc/dense_2/kernel/Adam_1Bvf/dense/biasBvf/dense/bias/AdamBvf/dense/bias/Adam_1Bvf/dense/kernelBvf/dense/kernel/AdamBvf/dense/kernel/Adam_1Bvf/dense_1/biasBvf/dense_1/bias/AdamBvf/dense_1/bias/Adam_1Bvf/dense_1/kernelBvf/dense_1/kernel/AdamBvf/dense_1/kernel/Adam_1Bvf/dense_2/biasBvf/dense_2/bias/AdamBvf/dense_2/bias/Adam_1Bvf/dense_2/kernelBvf/dense_2/kernel/AdamBvf/dense_2/kernel/Adam_1*
dtype0*
_output_shapes
:-
?
!save_1/RestoreV2/shape_and_slicesConst*
_output_shapes
:-*m
valuedBb-B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0
?
save_1/RestoreV2	RestoreV2save_1/Constsave_1/RestoreV2/tensor_names!save_1/RestoreV2/shape_and_slices*;
dtypes1
/2-*?
_output_shapes?
?:::::::::::::::::::::::::::::::::::::::::::::
?
save_1/AssignAssignbeta1_powersave_1/RestoreV2*
T0* 
_class
loc:@vc/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking(
?
save_1/Assign_1Assignbeta2_powersave_1/RestoreV2:1*
validate_shape(*
T0*
_output_shapes
: *
use_locking(* 
_class
loc:@vc/dense/bias
?
save_1/Assign_2Assignpi/dense/biassave_1/RestoreV2:2*
use_locking(*
T0*
validate_shape(* 
_class
loc:@pi/dense/bias*
_output_shapes	
:?
?
save_1/Assign_3Assignpi/dense/kernelsave_1/RestoreV2:3*
validate_shape(*
use_locking(*"
_class
loc:@pi/dense/kernel*
T0*
_output_shapes
:	<?
?
save_1/Assign_4Assignpi/dense_1/biassave_1/RestoreV2:4*
T0*"
_class
loc:@pi/dense_1/bias*
use_locking(*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_5Assignpi/dense_1/kernelsave_1/RestoreV2:5*
validate_shape(*
T0*
use_locking(* 
_output_shapes
:
??*$
_class
loc:@pi/dense_1/kernel
?
save_1/Assign_6Assignpi/dense_2/biassave_1/RestoreV2:6*
T0*
use_locking(*
validate_shape(*"
_class
loc:@pi/dense_2/bias*
_output_shapes
:
?
save_1/Assign_7Assignpi/dense_2/kernelsave_1/RestoreV2:7*
validate_shape(*$
_class
loc:@pi/dense_2/kernel*
_output_shapes
:	?*
use_locking(*
T0
?
save_1/Assign_8Assign
pi/log_stdsave_1/RestoreV2:8*
validate_shape(*
_class
loc:@pi/log_std*
use_locking(*
_output_shapes
:*
T0
?
save_1/Assign_9Assignvc/dense/biassave_1/RestoreV2:9*
_output_shapes	
:?* 
_class
loc:@vc/dense/bias*
T0*
use_locking(*
validate_shape(
?
save_1/Assign_10Assignvc/dense/bias/Adamsave_1/RestoreV2:10*
use_locking(* 
_class
loc:@vc/dense/bias*
T0*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_11Assignvc/dense/bias/Adam_1save_1/RestoreV2:11*
validate_shape(*
T0*
_output_shapes	
:?*
use_locking(* 
_class
loc:@vc/dense/bias
?
save_1/Assign_12Assignvc/dense/kernelsave_1/RestoreV2:12*
use_locking(*"
_class
loc:@vc/dense/kernel*
validate_shape(*
_output_shapes
:	<?*
T0
?
save_1/Assign_13Assignvc/dense/kernel/Adamsave_1/RestoreV2:13*"
_class
loc:@vc/dense/kernel*
validate_shape(*
T0*
use_locking(*
_output_shapes
:	<?
?
save_1/Assign_14Assignvc/dense/kernel/Adam_1save_1/RestoreV2:14*
validate_shape(*
use_locking(*
_output_shapes
:	<?*"
_class
loc:@vc/dense/kernel*
T0
?
save_1/Assign_15Assignvc/dense_1/biassave_1/RestoreV2:15*
T0*
use_locking(*
validate_shape(*
_output_shapes	
:?*"
_class
loc:@vc/dense_1/bias
?
save_1/Assign_16Assignvc/dense_1/bias/Adamsave_1/RestoreV2:16*
validate_shape(*
use_locking(*
T0*"
_class
loc:@vc/dense_1/bias*
_output_shapes	
:?
?
save_1/Assign_17Assignvc/dense_1/bias/Adam_1save_1/RestoreV2:17*
validate_shape(*
_output_shapes	
:?*
T0*"
_class
loc:@vc/dense_1/bias*
use_locking(
?
save_1/Assign_18Assignvc/dense_1/kernelsave_1/RestoreV2:18*
T0*
use_locking(*$
_class
loc:@vc/dense_1/kernel*
validate_shape(* 
_output_shapes
:
??
?
save_1/Assign_19Assignvc/dense_1/kernel/Adamsave_1/RestoreV2:19*$
_class
loc:@vc/dense_1/kernel* 
_output_shapes
:
??*
use_locking(*
validate_shape(*
T0
?
save_1/Assign_20Assignvc/dense_1/kernel/Adam_1save_1/RestoreV2:20*
use_locking(*
validate_shape(*
T0* 
_output_shapes
:
??*$
_class
loc:@vc/dense_1/kernel
?
save_1/Assign_21Assignvc/dense_2/biassave_1/RestoreV2:21*
validate_shape(*
_output_shapes
:*"
_class
loc:@vc/dense_2/bias*
use_locking(*
T0
?
save_1/Assign_22Assignvc/dense_2/bias/Adamsave_1/RestoreV2:22*
_output_shapes
:*"
_class
loc:@vc/dense_2/bias*
T0*
use_locking(*
validate_shape(
?
save_1/Assign_23Assignvc/dense_2/bias/Adam_1save_1/RestoreV2:23*
use_locking(*
T0*
_output_shapes
:*"
_class
loc:@vc/dense_2/bias*
validate_shape(
?
save_1/Assign_24Assignvc/dense_2/kernelsave_1/RestoreV2:24*
_output_shapes
:	?*$
_class
loc:@vc/dense_2/kernel*
use_locking(*
validate_shape(*
T0
?
save_1/Assign_25Assignvc/dense_2/kernel/Adamsave_1/RestoreV2:25*
T0*
_output_shapes
:	?*
use_locking(*
validate_shape(*$
_class
loc:@vc/dense_2/kernel
?
save_1/Assign_26Assignvc/dense_2/kernel/Adam_1save_1/RestoreV2:26*
T0*
_output_shapes
:	?*
validate_shape(*$
_class
loc:@vc/dense_2/kernel*
use_locking(
?
save_1/Assign_27Assignvf/dense/biassave_1/RestoreV2:27* 
_class
loc:@vf/dense/bias*
validate_shape(*
_output_shapes	
:?*
T0*
use_locking(
?
save_1/Assign_28Assignvf/dense/bias/Adamsave_1/RestoreV2:28* 
_class
loc:@vf/dense/bias*
T0*
use_locking(*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_29Assignvf/dense/bias/Adam_1save_1/RestoreV2:29*
use_locking(*
validate_shape(* 
_class
loc:@vf/dense/bias*
T0*
_output_shapes	
:?
?
save_1/Assign_30Assignvf/dense/kernelsave_1/RestoreV2:30*
T0*
use_locking(*
validate_shape(*
_output_shapes
:	<?*"
_class
loc:@vf/dense/kernel
?
save_1/Assign_31Assignvf/dense/kernel/Adamsave_1/RestoreV2:31*"
_class
loc:@vf/dense/kernel*
T0*
validate_shape(*
_output_shapes
:	<?*
use_locking(
?
save_1/Assign_32Assignvf/dense/kernel/Adam_1save_1/RestoreV2:32*
use_locking(*
T0*
validate_shape(*
_output_shapes
:	<?*"
_class
loc:@vf/dense/kernel
?
save_1/Assign_33Assignvf/dense_1/biassave_1/RestoreV2:33*
T0*
use_locking(*"
_class
loc:@vf/dense_1/bias*
validate_shape(*
_output_shapes	
:?
?
save_1/Assign_34Assignvf/dense_1/bias/Adamsave_1/RestoreV2:34*
_output_shapes	
:?*"
_class
loc:@vf/dense_1/bias*
validate_shape(*
T0*
use_locking(
?
save_1/Assign_35Assignvf/dense_1/bias/Adam_1save_1/RestoreV2:35*
T0*
_output_shapes	
:?*
validate_shape(*
use_locking(*"
_class
loc:@vf/dense_1/bias
?
save_1/Assign_36Assignvf/dense_1/kernelsave_1/RestoreV2:36*$
_class
loc:@vf/dense_1/kernel*
T0* 
_output_shapes
:
??*
use_locking(*
validate_shape(
?
save_1/Assign_37Assignvf/dense_1/kernel/Adamsave_1/RestoreV2:37* 
_output_shapes
:
??*
use_locking(*
T0*$
_class
loc:@vf/dense_1/kernel*
validate_shape(
?
save_1/Assign_38Assignvf/dense_1/kernel/Adam_1save_1/RestoreV2:38*$
_class
loc:@vf/dense_1/kernel*
T0*
use_locking(*
validate_shape(* 
_output_shapes
:
??
?
save_1/Assign_39Assignvf/dense_2/biassave_1/RestoreV2:39*
use_locking(*
T0*"
_class
loc:@vf/dense_2/bias*
_output_shapes
:*
validate_shape(
?
save_1/Assign_40Assignvf/dense_2/bias/Adamsave_1/RestoreV2:40*"
_class
loc:@vf/dense_2/bias*
validate_shape(*
T0*
_output_shapes
:*
use_locking(
?
save_1/Assign_41Assignvf/dense_2/bias/Adam_1save_1/RestoreV2:41*
validate_shape(*
T0*"
_class
loc:@vf/dense_2/bias*
use_locking(*
_output_shapes
:
?
save_1/Assign_42Assignvf/dense_2/kernelsave_1/RestoreV2:42*$
_class
loc:@vf/dense_2/kernel*
T0*
use_locking(*
_output_shapes
:	?*
validate_shape(
?
save_1/Assign_43Assignvf/dense_2/kernel/Adamsave_1/RestoreV2:43*
T0*
use_locking(*$
_class
loc:@vf/dense_2/kernel*
_output_shapes
:	?*
validate_shape(
?
save_1/Assign_44Assignvf/dense_2/kernel/Adam_1save_1/RestoreV2:44*
T0*$
_class
loc:@vf/dense_2/kernel*
use_locking(*
_output_shapes
:	?*
validate_shape(
?
save_1/restore_shardNoOp^save_1/Assign^save_1/Assign_1^save_1/Assign_10^save_1/Assign_11^save_1/Assign_12^save_1/Assign_13^save_1/Assign_14^save_1/Assign_15^save_1/Assign_16^save_1/Assign_17^save_1/Assign_18^save_1/Assign_19^save_1/Assign_2^save_1/Assign_20^save_1/Assign_21^save_1/Assign_22^save_1/Assign_23^save_1/Assign_24^save_1/Assign_25^save_1/Assign_26^save_1/Assign_27^save_1/Assign_28^save_1/Assign_29^save_1/Assign_3^save_1/Assign_30^save_1/Assign_31^save_1/Assign_32^save_1/Assign_33^save_1/Assign_34^save_1/Assign_35^save_1/Assign_36^save_1/Assign_37^save_1/Assign_38^save_1/Assign_39^save_1/Assign_4^save_1/Assign_40^save_1/Assign_41^save_1/Assign_42^save_1/Assign_43^save_1/Assign_44^save_1/Assign_5^save_1/Assign_6^save_1/Assign_7^save_1/Assign_8^save_1/Assign_9
1
save_1/restore_allNoOp^save_1/restore_shard "B
save_1/Const:0save_1/Identity:0save_1/restore_all (5 @F8"
train_op

Adam"?
trainable_variables??
s
pi/dense/kernel:0pi/dense/kernel/Assignpi/dense/kernel/read:02,pi/dense/kernel/Initializer/random_uniform:08
b
pi/dense/bias:0pi/dense/bias/Assignpi/dense/bias/read:02!pi/dense/bias/Initializer/zeros:08
{
pi/dense_1/kernel:0pi/dense_1/kernel/Assignpi/dense_1/kernel/read:02.pi/dense_1/kernel/Initializer/random_uniform:08
j
pi/dense_1/bias:0pi/dense_1/bias/Assignpi/dense_1/bias/read:02#pi/dense_1/bias/Initializer/zeros:08
{
pi/dense_2/kernel:0pi/dense_2/kernel/Assignpi/dense_2/kernel/read:02.pi/dense_2/kernel/Initializer/random_uniform:08
j
pi/dense_2/bias:0pi/dense_2/bias/Assignpi/dense_2/bias/read:02#pi/dense_2/bias/Initializer/zeros:08
R
pi/log_std:0pi/log_std/Assignpi/log_std/read:02pi/log_std/initial_value:08
s
vf/dense/kernel:0vf/dense/kernel/Assignvf/dense/kernel/read:02,vf/dense/kernel/Initializer/random_uniform:08
b
vf/dense/bias:0vf/dense/bias/Assignvf/dense/bias/read:02!vf/dense/bias/Initializer/zeros:08
{
vf/dense_1/kernel:0vf/dense_1/kernel/Assignvf/dense_1/kernel/read:02.vf/dense_1/kernel/Initializer/random_uniform:08
j
vf/dense_1/bias:0vf/dense_1/bias/Assignvf/dense_1/bias/read:02#vf/dense_1/bias/Initializer/zeros:08
{
vf/dense_2/kernel:0vf/dense_2/kernel/Assignvf/dense_2/kernel/read:02.vf/dense_2/kernel/Initializer/random_uniform:08
j
vf/dense_2/bias:0vf/dense_2/bias/Assignvf/dense_2/bias/read:02#vf/dense_2/bias/Initializer/zeros:08
s
vc/dense/kernel:0vc/dense/kernel/Assignvc/dense/kernel/read:02,vc/dense/kernel/Initializer/random_uniform:08
b
vc/dense/bias:0vc/dense/bias/Assignvc/dense/bias/read:02!vc/dense/bias/Initializer/zeros:08
{
vc/dense_1/kernel:0vc/dense_1/kernel/Assignvc/dense_1/kernel/read:02.vc/dense_1/kernel/Initializer/random_uniform:08
j
vc/dense_1/bias:0vc/dense_1/bias/Assignvc/dense_1/bias/read:02#vc/dense_1/bias/Initializer/zeros:08
{
vc/dense_2/kernel:0vc/dense_2/kernel/Assignvc/dense_2/kernel/read:02.vc/dense_2/kernel/Initializer/random_uniform:08
j
vc/dense_2/bias:0vc/dense_2/bias/Assignvc/dense_2/bias/read:02#vc/dense_2/bias/Initializer/zeros:08"?*
	variables?*?*
s
pi/dense/kernel:0pi/dense/kernel/Assignpi/dense/kernel/read:02,pi/dense/kernel/Initializer/random_uniform:08
b
pi/dense/bias:0pi/dense/bias/Assignpi/dense/bias/read:02!pi/dense/bias/Initializer/zeros:08
{
pi/dense_1/kernel:0pi/dense_1/kernel/Assignpi/dense_1/kernel/read:02.pi/dense_1/kernel/Initializer/random_uniform:08
j
pi/dense_1/bias:0pi/dense_1/bias/Assignpi/dense_1/bias/read:02#pi/dense_1/bias/Initializer/zeros:08
{
pi/dense_2/kernel:0pi/dense_2/kernel/Assignpi/dense_2/kernel/read:02.pi/dense_2/kernel/Initializer/random_uniform:08
j
pi/dense_2/bias:0pi/dense_2/bias/Assignpi/dense_2/bias/read:02#pi/dense_2/bias/Initializer/zeros:08
R
pi/log_std:0pi/log_std/Assignpi/log_std/read:02pi/log_std/initial_value:08
s
vf/dense/kernel:0vf/dense/kernel/Assignvf/dense/kernel/read:02,vf/dense/kernel/Initializer/random_uniform:08
b
vf/dense/bias:0vf/dense/bias/Assignvf/dense/bias/read:02!vf/dense/bias/Initializer/zeros:08
{
vf/dense_1/kernel:0vf/dense_1/kernel/Assignvf/dense_1/kernel/read:02.vf/dense_1/kernel/Initializer/random_uniform:08
j
vf/dense_1/bias:0vf/dense_1/bias/Assignvf/dense_1/bias/read:02#vf/dense_1/bias/Initializer/zeros:08
{
vf/dense_2/kernel:0vf/dense_2/kernel/Assignvf/dense_2/kernel/read:02.vf/dense_2/kernel/Initializer/random_uniform:08
j
vf/dense_2/bias:0vf/dense_2/bias/Assignvf/dense_2/bias/read:02#vf/dense_2/bias/Initializer/zeros:08
s
vc/dense/kernel:0vc/dense/kernel/Assignvc/dense/kernel/read:02,vc/dense/kernel/Initializer/random_uniform:08
b
vc/dense/bias:0vc/dense/bias/Assignvc/dense/bias/read:02!vc/dense/bias/Initializer/zeros:08
{
vc/dense_1/kernel:0vc/dense_1/kernel/Assignvc/dense_1/kernel/read:02.vc/dense_1/kernel/Initializer/random_uniform:08
j
vc/dense_1/bias:0vc/dense_1/bias/Assignvc/dense_1/bias/read:02#vc/dense_1/bias/Initializer/zeros:08
{
vc/dense_2/kernel:0vc/dense_2/kernel/Assignvc/dense_2/kernel/read:02.vc/dense_2/kernel/Initializer/random_uniform:08
j
vc/dense_2/bias:0vc/dense_2/bias/Assignvc/dense_2/bias/read:02#vc/dense_2/bias/Initializer/zeros:08
T
beta1_power:0beta1_power/Assignbeta1_power/read:02beta1_power/initial_value:0
T
beta2_power:0beta2_power/Assignbeta2_power/read:02beta2_power/initial_value:0
|
vf/dense/kernel/Adam:0vf/dense/kernel/Adam/Assignvf/dense/kernel/Adam/read:02(vf/dense/kernel/Adam/Initializer/zeros:0
?
vf/dense/kernel/Adam_1:0vf/dense/kernel/Adam_1/Assignvf/dense/kernel/Adam_1/read:02*vf/dense/kernel/Adam_1/Initializer/zeros:0
t
vf/dense/bias/Adam:0vf/dense/bias/Adam/Assignvf/dense/bias/Adam/read:02&vf/dense/bias/Adam/Initializer/zeros:0
|
vf/dense/bias/Adam_1:0vf/dense/bias/Adam_1/Assignvf/dense/bias/Adam_1/read:02(vf/dense/bias/Adam_1/Initializer/zeros:0
?
vf/dense_1/kernel/Adam:0vf/dense_1/kernel/Adam/Assignvf/dense_1/kernel/Adam/read:02*vf/dense_1/kernel/Adam/Initializer/zeros:0
?
vf/dense_1/kernel/Adam_1:0vf/dense_1/kernel/Adam_1/Assignvf/dense_1/kernel/Adam_1/read:02,vf/dense_1/kernel/Adam_1/Initializer/zeros:0
|
vf/dense_1/bias/Adam:0vf/dense_1/bias/Adam/Assignvf/dense_1/bias/Adam/read:02(vf/dense_1/bias/Adam/Initializer/zeros:0
?
vf/dense_1/bias/Adam_1:0vf/dense_1/bias/Adam_1/Assignvf/dense_1/bias/Adam_1/read:02*vf/dense_1/bias/Adam_1/Initializer/zeros:0
?
vf/dense_2/kernel/Adam:0vf/dense_2/kernel/Adam/Assignvf/dense_2/kernel/Adam/read:02*vf/dense_2/kernel/Adam/Initializer/zeros:0
?
vf/dense_2/kernel/Adam_1:0vf/dense_2/kernel/Adam_1/Assignvf/dense_2/kernel/Adam_1/read:02,vf/dense_2/kernel/Adam_1/Initializer/zeros:0
|
vf/dense_2/bias/Adam:0vf/dense_2/bias/Adam/Assignvf/dense_2/bias/Adam/read:02(vf/dense_2/bias/Adam/Initializer/zeros:0
?
vf/dense_2/bias/Adam_1:0vf/dense_2/bias/Adam_1/Assignvf/dense_2/bias/Adam_1/read:02*vf/dense_2/bias/Adam_1/Initializer/zeros:0
|
vc/dense/kernel/Adam:0vc/dense/kernel/Adam/Assignvc/dense/kernel/Adam/read:02(vc/dense/kernel/Adam/Initializer/zeros:0
?
vc/dense/kernel/Adam_1:0vc/dense/kernel/Adam_1/Assignvc/dense/kernel/Adam_1/read:02*vc/dense/kernel/Adam_1/Initializer/zeros:0
t
vc/dense/bias/Adam:0vc/dense/bias/Adam/Assignvc/dense/bias/Adam/read:02&vc/dense/bias/Adam/Initializer/zeros:0
|
vc/dense/bias/Adam_1:0vc/dense/bias/Adam_1/Assignvc/dense/bias/Adam_1/read:02(vc/dense/bias/Adam_1/Initializer/zeros:0
?
vc/dense_1/kernel/Adam:0vc/dense_1/kernel/Adam/Assignvc/dense_1/kernel/Adam/read:02*vc/dense_1/kernel/Adam/Initializer/zeros:0
?
vc/dense_1/kernel/Adam_1:0vc/dense_1/kernel/Adam_1/Assignvc/dense_1/kernel/Adam_1/read:02,vc/dense_1/kernel/Adam_1/Initializer/zeros:0
|
vc/dense_1/bias/Adam:0vc/dense_1/bias/Adam/Assignvc/dense_1/bias/Adam/read:02(vc/dense_1/bias/Adam/Initializer/zeros:0
?
vc/dense_1/bias/Adam_1:0vc/dense_1/bias/Adam_1/Assignvc/dense_1/bias/Adam_1/read:02*vc/dense_1/bias/Adam_1/Initializer/zeros:0
?
vc/dense_2/kernel/Adam:0vc/dense_2/kernel/Adam/Assignvc/dense_2/kernel/Adam/read:02*vc/dense_2/kernel/Adam/Initializer/zeros:0
?
vc/dense_2/kernel/Adam_1:0vc/dense_2/kernel/Adam_1/Assignvc/dense_2/kernel/Adam_1/read:02,vc/dense_2/kernel/Adam_1/Initializer/zeros:0
|
vc/dense_2/bias/Adam:0vc/dense_2/bias/Adam/Assignvc/dense_2/bias/Adam/read:02(vc/dense_2/bias/Adam/Initializer/zeros:0
?
vc/dense_2/bias/Adam_1:0vc/dense_2/bias/Adam_1/Assignvc/dense_2/bias/Adam_1/read:02*vc/dense_2/bias/Adam_1/Initializer/zeros:0*?
serving_default?
)
x$
Placeholder:0?????????<%
pi
pi/add:0?????????%
vc
vc/Squeeze:0?????????$
v
vf/Squeeze:0?????????tensorflow/serving/predict