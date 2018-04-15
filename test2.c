/* To Build
   g++ -O3 -std=c++11 -mavx2 -I../../include/ test.c utils.cpp -o test
*/

#include <immintrin.h> /* AVX intrinsics */
#include "avxcommon.h"
#include <iostream>
//#include "avxsort_core.h"
#include <cstdio>
#include "utils.h"
using namespace std;
inline void __attribute__((always_inline))
merge16_varlen(int64_t * restrict inpA, int64_t * restrict inpAv,
	int64_t * restrict inpB, int64_t * restrict inpBv,
	int64_t * restrict Out, int64_t * restrict Outv,
	const uint32_t lenA, const uint32_t lenB);

#define LENA 5000
#define LENB 5000
void merge16_varlen(int64_t * restrict inpA, int64_t * restrict inpAv,
	int64_t * restrict inpB, int64_t * restrict inpBv,
	int64_t * restrict Out, int64_t * restrict Outv,
	const uint32_t lenA, const uint32_t lenB);
void test_merge16_varlen(void)
{
	int64_t inA[LENA];
	int64_t inAv[LENA];
	int64_t inB[LENB];
	int64_t inBv[LENB];
	int ten = 16;
	inA[0] = 0;
	inB[0] = 2;
	inAv[0] = 0;
	inBv[0] = 2;
	for (int i = 1; i < LENA; i++)
	{
		inA[i] = i;
		inB[i] = 2*i;
		inAv[i] = inAv[i-1]+rand()%10;
		inBv[i] = inBv[i-1]+rand()%10;
	}
	int64_t out[LENA + LENB];
	int64_t outv[LENA+LENB];
	merge16_varlen(inA, inAv, inB, inBv, out, outv, LENA, LENB);
	for(int i = 0; i < LENA+LENB-1;i++)
	{
		if(outv[i]==outv[i+1])
		{
			if(out[i]>out[i+1])
			{
				printf("ERROR");
				break;
			}
		}
	}
}

void
avxsort_unaligned(int64_t ** inputptr, int64_t ** inputptrv, 
                  int64_t ** outputptr, int64_t ** outputptrv, uint64_t nitems);

#define NUM_ITEMS (1024)  /* 2 * L2_CACHE_SIZE */
#define ITEMS 4
int main(int argc, char *argv[])
{
	/*
	//test_merge16_varlen();
	double const * address_values;
	double const * address_keys;
	double const values[20] = { 1,2,3,4,4,3,2,1,1,1,1,1,1,1,1,1,1,1,1,1 };
	double const keys[20] = { 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 };
	address_values = values;
	address_keys = keys;
	__m256d lower_values;
	__m256d upper_values;
	__m256d lower_keys;
	__m256d upper_keys;
	__m256d O1;
	__m256d O1v;
	__m256d O2;
	__m256d O2v;
	
	LOAD8U(lower_values, upper_values, address_values);
	LOAD8U(lower_keys, upper_keys, address_keys);
	
	printf("UPPER VALUES\n");
	printf("%f %f %f %f\n", upper_values[0], upper_values[1], upper_values[2], upper_values[3]);
	printf("\nLOWER VALUES\n");
	printf("%f %f %f %f\n", lower_values[0], lower_values[1], lower_values[2], lower_values[3]);
	

	
	printf("\nO1\n");
	printf("%f %f %f %f\n", O1[0], O1[1], O1[2], O1[3]);
	printf("\nO2\n");
	printf("%f %f %f %f\n", O2[0], O2[1], O2[2], O2[3]);
	printf("\nO1v\n");
	printf("%f %f %f %f\n", O1v[0], O1v[1], O1v[2], O1v[3]);
	printf("\nO2v\n");
	printf("%f %f %f %f\n", O2v[0], O2v[1], O2v[2], O2v[3]);
	*/
	/*
	double const * address_values;
	double const * address_keys;
	double const values[8] = { 10,10,23,34,10,10,98,100 };
	double const keys[8] = { 1,2,3,4,5,6,7,8};
	address_values = values;
	address_keys = keys;
	__m256d vector1;
	__m256d vector2;
	__m256d vector1_keys;
	__m256d vector2_keys;
	__m256d output1;
	__m256d output1v;
	__m256d output2;
	__m256d output2v;
	//load first eight values to vectors
	LOAD8U(vector1, vector2, address_values);
	LOAD8U(vector1_keys, vector2_keys, address_keys);
	
	printf("VECTOR1: \n");
	printf("%f %f %f %f\n\n", vector1[0], vector1[1], vector1[2], vector1[3]);
	printf("VECTOR1 KEYS: \n");
	printf("%f %f %f %f\n\n", vector1_keys[0], vector1_keys[1], vector1_keys[2], vector1_keys[3]);
	printf("VECTOR2:\n");
	printf("%f %f %f %f\n\n", vector2[0], vector2[1], vector2[2], vector2[3]);
	printf("VECTOR2 KEYS:\n");
	printf("%f %f %f %f\n\n", vector2_keys[0], vector2_keys[1], vector2_keys[2], vector2_keys[3]);
	printf("\n\n");
	printf("ASDFASDFDASFD");
	BITONIC_MERGE4(output1, output1v, output2, output2v, vector1,vector1_keys, vector2, vector2_keys);
	printf("\n\n");
	printf("OUTPUT1: \n");
	printf("%f %f %f %f\n\n", output1[0], output1[1], output1[2], output1[3]);
	printf("OUTPUT1_KEYS:\n");
	printf("%f %f %f %f\n\n", output1v[0], output1v[1], output1v[2], output1v[3]);
	printf("OUTPUT2: \n");
	printf("%f %f %f %f\n\n", output2[0], output2[1], output2[2], output2[3]);
	printf("OUTPUT2 KEYS:\n");
	printf("%f %f %f %f\n\n", output2v[0], output2v[1], output2v[2], output2v[3]);
	*/
	int64_t *inA, *inAv, *outA, *outAv;
	inA = (int64_t*)malloc(NUM_ITEMS * sizeof(int64_t));
	inAv = (int64_t*)malloc(NUM_ITEMS * sizeof(int64_t));
	outA = (int64_t*)malloc(NUM_ITEMS * sizeof(int64_t));
	outAv = (int64_t*)malloc(NUM_ITEMS * sizeof(int64_t));
	for (int i = 0; i < NUM_ITEMS; i++)
	{
		inA[i] = rand();
		inAv[i] = i;
	}
	avxsort_unaligned(&inA,&inAv,&outA,&outAv,NUM_ITEMS);	
	free(inA);
	free(inAv);
	free(outA);
	free(outAv);
	//test_merge16_varlen();
    return 0;
}

inline void __attribute__((always_inline))
merge16_varlen(int64_t * restrict inpA, int64_t * restrict inpAv,
	int64_t * restrict inpB, int64_t * restrict inpBv,
	int64_t * restrict Out, int64_t * restrict Outv,
	const uint32_t lenA, const uint32_t lenB)
{
	//set lenA16 and lenB16 as lenA and lenB except we set the last four bits to zero
	//this results in a number that is divisible by 16
	uint32_t lenA16 = lenA & ~0xF, lenB16 = lenB & ~0xF;

	//set uint32_t initial values to zero
	uint32_t ai = 0, bi = 0;

	//set pointers to output and outputv
	int64_t * out = Out;
	int64_t * outv = Outv;

	//the first case is the most common where there is greater than 16 numbers to be sorted
	if (lenA16 > 16 && lenB16 > 16) {
		//we create multiple blocks that are to be kept inside the register since we'll be using them a lot
		//what's contained in a block is a val[16] meaning each block stores up to 16 values
		//each lines is a block pointing to a cased inpA inpB inpAv and inpBv
		register block16 * inA = (block16 *)inpA;
		register block16 * inB = (block16 *)inpB;
		register block16 * inAv = (block16 *)inpAv;
		register block16 * inBv = (block16 *)inpBv;

		printf("YES\n");

		//this is the last block to be sorted
		block16 * const    endA = (block16 *)(inpA + lenA) - 1;
		block16 * const    endB = (block16 *)(inpB + lenB) - 1;
		
		//these are the output variables
		block16 * outp = (block16 *)out;
		block16 * outpv = (block16 *)outv;


		//not sure what this does
		register block16 * next = inB;
		register block16 * nextv = inBv;


		//declare multiple avx registers (each of them stores up to 8 doubles)
		__m256d outreg1l1, outreg1l2, outreg1h1, outreg1h2;
		__m256d outreg2l1, outreg2l2, outreg2h1, outreg2h2;
		__m256d outreg1l1v, outreg1l2v, outreg1h1v, outreg1h2v;
		__m256d outreg2l1v, outreg2l2v, outreg2h1v, outreg2h2v;

		__m256d regAl1, regAl2, regAh1, regAh2;
		__m256d regBl1, regBl2, regBh1, regBh2;
		__m256d regAl1v, regAl2v, regAh1v, regAh2v;
		__m256d regBl1v, regBl2v, regBh1v, regBh2v;


		//load 2 avx 256 bit registers from inA
		LOAD8U(regAl1, regAl2, inA);

		//load 2 avx 256 bit registers from the next block inA
		LOAD8U(regAh1, regAh2, ((block8 *)(inA)+1));

		//increment inA
		inA++;

		//load 2 avx 256 bit registers from inAv and inAv+1
		LOAD8U(regAl1v, regAl2v, inAv);
		LOAD8U(regAh1v, regAh2v, ((block8 *)(inAv)+1));

		//increment inAv
		inAv++;

		//load 2 avx 256 bit registers from inB and inB+1
		LOAD8U(regBl1, regBl2, inB);
		LOAD8U(regBh1, regBh2, ((block8 *)(inB)+1));
		inB++;

		//load 2 avx 256 bit registers from inBv and inBv+1
		LOAD8U(regBl1v, regBl2v, inBv);
		LOAD8U(regBh1v, regBh2v, ((block8 *)(inBv)+1));
		inBv++;

		//what happens next is we perform a bitonic merge with the loaded register values
		BITONIC_MERGE16(outreg1l1, outreg1l1v, outreg1l2, outreg1l2v,
			outreg1h1, outreg1h1v, outreg1h2, outreg1h2v,
			outreg2l1, outreg2l1v, outreg2l2, outreg2l2v,
			outreg2h1, outreg2h1v, outreg2h2, outreg2h2v,
			regAl1, regAl1v, regAl2, regAl2v,
			regAh1, regAh1v, regAh2, regAh2v,
			regBl1, regBl1v, regBl2, regBl2v,
			regBh1, regBh1v, regBh2, regBh2v);

		//we store the 8 unsigned registers out reg lower and out reg lower into outp
		STORE8U(outp, outreg1l1, outreg1l2);

		//store 8 unsigned registers into the upper portion of output (upper portion)
		STORE8U(((block8 *)outp + 1), outreg1h1, outreg1h2);
		outp++;

		//do the exact same thing for the keys
		STORE8U(outpv, outreg1l1v, outreg1l2v);
		STORE8U(((block8 *)outpv + 1), outreg1h1v, outreg1h2v);

		//increment by 1 aka 16
		outpv++;


		//now we keep on looping up until we reach the end of array of endA and endB
		while (inA < endA && inB < endB) {

			/** The inline assembly below does exactly the following code: */
			/* Option 3: with assembly */
			//not sure what this exactly does, the more I looked at it the more confused I get
			IFELSECONDMOVE(next, nextv, inA, inAv, inB, inBv, 128);

			//update the registers with whatever was in the output
			regAl1 = outreg2l1;
			regAl2 = outreg2l2;
			regAh1 = outreg2h1;
			regAh2 = outreg2h2;
			regAl1v = outreg2l1v;
			regAl2v = outreg2l2v;
			regAh1v = outreg2h1v;
			regAh2v = outreg2h2v;

			//load 256 bits in lower 1 and 2 from next
			LOAD8U(regBl1, regBl2, next);

			//load the other 256 bits from next + 1 (upper side)
			LOAD8U(regBh1, regBh2, ((block8 *)next + 1));

			//load 256 bits from lower 1 and 2 from the keys
			LOAD8U(regBl1v, regBl2v, nextv);

			//load 256 bits from higher 1 and 2 from the keys (next + 1)
			LOAD8U(regBh1v, regBh2v, ((block8 *)nextv + 1));

			//start a bitonic merge network for 32 elements
			BITONIC_MERGE16(outreg1l1, outreg1l1v, outreg1l2, outreg1l2v,
				outreg1h1, outreg1h1v, outreg1h2, outreg1h2v,
				outreg2l1, outreg2l1v, outreg2l2, outreg2l2v,
				outreg2h1, outreg2h1v, outreg2h2, outreg2h2v,
				regAl1, regAl1v, regAl2, regAl2v,
				regAh1, regAh1v, regAh2, regAh2v,
				regBl1, regBl1v, regBl2, regBl2v,
				regBh1, regBh1v, regBh2, regBh2v);

			/* store outreg1 */
			//store the results from the lower two registers into the output register
			STORE8U(outp, outreg1l1, outreg1l2);
			//store the results from the upper two registers into the upper register
			STORE8U(((block8 *)outp + 1), outreg1h1, outreg1h2);

			//increment outp
			outp++;

			//same thing
			STORE8U(outpv, outreg1l1v, outreg1l2v);
			STORE8U(((block8 *)outpv + 1), outreg1h1v, outreg1h2v);
			//increment outpv (contains the address of out and or outv
			outpv++;
		}

		//not sure what this exactly does
		/* flush the register to one of the lists */
		int64_t hireg[4] __attribute__((aligned(32)));
		_mm256_store_pd((double *)hireg, outreg2h2);

		if (*((int64_t *)inA) >= *((int64_t*)(hireg + 3))) {
			/* store the last remaining register values to A */
			inA--;
			STORE8U(inA, outreg2l1, outreg2l2);
			STORE8U(((block8 *)inA + 1), outreg2h1, outreg2h2);
			inAv--;
			STORE8U(inAv, outreg2l1v, outreg2l2v);
			STORE8U(((block8 *)inAv + 1), outreg2h1v, outreg2h2v);
		}
		else {
			/* store the last remaining register values to B */
			inB--;
			STORE8U(inB, outreg2l1, outreg2l2);
			STORE8U(((block8 *)inB + 1), outreg2h1, outreg2h2);
			inBv--;
			STORE8U(inBv, outreg2l1v, outreg2l2v);
			STORE8U(((block8 *)inBv + 1), outreg2h1v, outreg2h2v);
		}

		//intitialize the loop control variables (remaining values between 0000 and 1111
		ai = ((int64_t *)inA - inpA);
		bi = ((int64_t *)inB - inpB);

		inpA = (int64_t *)inA;
		inpB = (int64_t *)inB;
		out = (int64_t *)outp;
		inpAv = (int64_t *)inAv;
		inpBv = (int64_t *)inBv;
		outv = (int64_t *)outpv;
	}
	/* serial-merge */
	//while the last set of values are less than the total length of either a and/or b
	while (ai < lenA && bi < lenB) {

		//create comparisons with the last set of cariables (A and B)
		int64_t * in = inpB;
		int64_t * inv = inpBv;
		uint32_t cmp = (*inpA < *inpB);
		uint32_t notcmp = !cmp;

		ai += cmp;
		bi += notcmp;

		if (cmp) {
			in = inpA;
			inv = inpAv;
		}
		*out = *in;
		*outv = *inv;
		out++;
		outv++;
		inpA += cmp;
		inpB += notcmp;
		inpAv += cmp;
		inpBv += notcmp;
	}

	//situation in which there are still a variables needing to be processed
	if (ai < lenA) {
		/* if A has any more items to be output */

		if ((lenA - ai) >= 8) {
			/* if A still has some times to be output with AVX */
			uint32_t lenA8 = ((lenA - ai) & ~0x7);
			register block8 * inA = (block8 *)inpA;
			block8 * const    endA = (block8 *)(inpA + lenA8);
			block8 * outp = (block8 *)out;
			register block8 * inAv = (block8 *)inpAv;
			block8 * outpv = (block8 *)outv;

			while (inA < endA) {
				__m256d regAl, regAh;
				__m256d regAlv, regAhv;

				LOAD8U(regAl, regAh, inA);
				STORE8U(outp, regAl, regAh);
				LOAD8U(regAlv, regAhv, inAv);
				STORE8U(outpv, regAlv, regAhv);

				outp++;
				inA++;
				outpv++;
				inAv++;
			}

			ai += ((int64_t*)inA - inpA);
			inpA = (int64_t *)inA;
			out = (int64_t *)outp;
			inpAv = (int64_t *)inAv;
			outv = (int64_t *)outpv;
		}

		while (ai < lenA) {
			*out = *inpA;
			ai++;
			out++;
			inpA++;
			*outv = *inpAv;
			outv++;
			inpAv++;

		}
	}
	//situation in which there are still b items to be processed
	else if (bi < lenB) {
		/* if B has any more items to be output */

		if ((lenB - bi) >= 8) {
			/* if B still has some times to be output with AVX */
			uint32_t lenB8 = ((lenB - bi) & ~0x7);
			register block8 * inB = (block8 *)inpB;
			block8 * const    endB = (block8 *)(inpB + lenB8);
			block8 * outp = (block8 *)out;
			register block8 * inBv = (block8 *)inpBv;
			block8 * outpv = (block8 *)outv;

			while (inB < endB) {
				__m256d regBl, regBh;
				__m256d regBlv, regBhv;
				LOAD8U(regBl, regBh, inB);
				STORE8U(outp, regBl, regBh);
				LOAD8U(regBlv, regBhv, inBv);
				STORE8U(outpv, regBlv, regBhv);
				outp++;
				inB++;
				outpv++;
				inBv++;
			}

			bi += ((int64_t*)inB - inpB);
			inpB = (int64_t *)inB;
			out = (int64_t *)outp;
			inpBv = (int64_t *)inBv;
			outv = (int64_t *)outpv;
		}

		while (bi < lenB) {
			*out = *inpB;
			*outv = *inpBv;
			bi++;
			out++;
			inpB++;
			outv++;
			inpBv++;
		}
	}
}


void
avxsort_unaligned(int64_t ** inputptr, int64_t ** inputptrv,
	int64_t ** outputptr, int64_t ** outputptrv, uint64_t nitems)
{
	if (nitems <= 0)
		return;
	// dump_arr_int64("input", *inputptr, nitems);

	int64_t * input = *inputptr;
	int64_t * output = *outputptr;
	int64_t * inputv = *inputptrv;
	int64_t * outputv = *outputptrv;

	uint64_t i;
	uint64_t nchunks = (nitems / BLOCKSIZE);
	int rem = (nitems % BLOCKSIZE);

	/* each chunk keeps track of its temporary memory offset */
	int64_t * ptrs[nchunks + 1][2];/* [chunk-in, chunk-out-tmp] */
	int64_t * ptrsv[nchunks + 1][2];/* [chunk-in, chunk-out-tmp] */
	uint32_t sizes[nchunks + 1];

	for (i = 0; i <= nchunks; i++) {
		ptrs[i][0] = input + i * BLOCKSIZE;
		ptrsv[i][0] = inputv + i * BLOCKSIZE;
		ptrs[i][1] = output + i * BLOCKSIZE;
		ptrsv[i][1] = outputv + i * BLOCKSIZE;
		sizes[i] = BLOCKSIZE;
	}

	/** 1) Divide the input into chunks fitting into L2 cache. */
	/* one more chunk if not divisible */
	for (i = 0; i < nchunks; i++) {
		avxsort_block(&ptrs[i][0], &ptrsv[i][0], &ptrs[i][1], &ptrsv[i][1], BLOCKSIZE);
		// dump_arr_int64("check 0", ptrs[i][1], BLOCKSIZE);
		swap(&ptrs[i][0], &ptrs[i][1]);
		swap(&ptrsv[i][0], &ptrsv[i][1]);
	}

	if (rem) {
		// xzl_bug_on(1);
		/* sort the last chunk which is less than BLOCKSIZE */
		avxsort_rem(&ptrs[i][0], &ptrsv[i][0], &ptrs[i][1], &ptrsv[i][1], rem);
		// dump_arr_int64("check 1", ptrs[i][1], rem);
		swap(&ptrs[i][0], &ptrs[i][1]);
		swap(&ptrsv[i][0], &ptrsv[i][1]);
		sizes[i] = rem;
	}


	/**
	* 2.a) for itr = [(logM) .. (logN -1)], merge sequences of length 2^itr to
	* obtain sorted sequences of length 2^{itr+1}.
	*/
	nchunks += (rem > 0);
	/* printf("Merge chunks = %d\n", nchunks); */
	const uint64_t logN = ceil(log2(nitems));
	for (i = LOG2_BLOCKSIZE; i < logN; i++) {

		uint64_t k = 0;
		for (uint64_t j = 0; j < (nchunks - 1); j += 2) {
			int64_t * inpA = ptrs[j][0];
			int64_t * inpB = ptrs[j + 1][0];
			int64_t * out = ptrs[j][1];
			int64_t * inpAv = ptrsv[j][0];
			int64_t * inpBv = ptrsv[j + 1][0];
			int64_t * outv = ptrsv[j][1];

			uint32_t  sizeA = sizes[j];
			uint32_t  sizeB = sizes[j + 1];

			/* need to change */
			merge16_varlen(inpA, inpAv, inpB, inpBv, out, outv, sizeA, sizeB);

			/* setup new pointers */
			ptrs[k][0] = out;
			ptrs[k][1] = inpA;
			ptrsv[k][0] = outv;
			ptrsv[k][1] = inpAv;

			sizes[k] = sizeA + sizeB;
			k++;
		}

		if ((nchunks % 2)) {
			/* just move the pointers */
			ptrs[k][0] = ptrs[nchunks - 1][0];
			ptrs[k][1] = ptrs[nchunks - 1][1];
			ptrsv[k][0] = ptrsv[nchunks - 1][0];
			ptrsv[k][1] = ptrsv[nchunks - 1][1];

			sizes[k] = sizes[nchunks - 1];
			k++;
		}

		nchunks = k;
	}

	/* finally swap input/output pointers, where output holds the sorted list */
	*outputptr = ptrs[0][0];
	*inputptr = ptrs[0][1];
	*outputptrv = ptrsv[0][0];
	*inputptrv = ptrsv[0][1];

}
*/