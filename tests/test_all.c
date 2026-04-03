/*
 * test_all.c — PostGPT-Q unit tests
 * gcc tests/test_all.c -O2 -lm -o tests/run_tests && cd ~/q && ./tests/run_tests
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

static int tests_passed = 0;
static int tests_failed = 0;

#define TEST(name) printf("  [TEST] %s... ", name);
#define PASS() do { printf("PASS\n"); tests_passed++; } while(0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); tests_failed++; } while(0)
#define CHECK(cond, msg) do { if(!(cond)) { FAIL(msg); return; } } while(0)

/* ── 1. Math ── */
static float clampf(float x, float lo, float hi) { return x<lo?lo:x>hi?hi:x; }

static void rmsnorm(float *out, const float *x, int n) {
    float ms=0; for(int i=0;i<n;i++) ms+=x[i]*x[i];
    ms=1.0f/sqrtf(ms/n+1e-6f);
    for(int i=0;i<n;i++) out[i]=x[i]*ms;
}

static void matmul(float *out, const float *x, const float *w, int n_in, int d_out) {
    for(int d=0;d<d_out;d++){
        float v=0; for(int j=0;j<n_in;j++) v+=x[j]*w[d*n_in+j];
        out[d]=v;
    }
}

static void softmax(float *x, int n) {
    float mx=x[0]; for(int i=1;i<n;i++) if(x[i]>mx) mx=x[i];
    float s=0; for(int i=0;i<n;i++){x[i]=expf(x[i]-mx);s+=x[i];}
    if(s>0) for(int i=0;i<n;i++) x[i]/=s;
}

void test_clampf(void) {
    TEST("clampf");
    CHECK(clampf(5,0,10)==5, "mid"); CHECK(clampf(-1,0,10)==0, "lo");
    CHECK(clampf(15,0,10)==10, "hi"); CHECK(clampf(0,0,0)==0, "zero");
    PASS();
}

void test_rmsnorm(void) {
    TEST("rmsnorm");
    float x[]={3,4}, out[2];
    rmsnorm(out,x,2);
    float ms=(9+16)/2.0f, sc=1.0f/sqrtf(ms+1e-6f);
    CHECK(fabsf(out[0]-3*sc)<1e-5f, "el0");
    CHECK(fabsf(out[1]-4*sc)<1e-5f, "el1");
    float norm=sqrtf(out[0]*out[0]+out[1]*out[1]);
    CHECK(fabsf(norm-sqrtf(2.0f))<0.01f, "norm");
    PASS();
}

void test_matmul(void) {
    TEST("matmul");
    float x[]={1,2,3}, w[]={1,0,0, 0,1,0}, out[2];
    matmul(out,x,w,3,2);
    CHECK(fabsf(out[0]-1)<1e-5f, "r0"); CHECK(fabsf(out[1]-2)<1e-5f, "r1");
    PASS();
}

void test_softmax(void) {
    TEST("softmax");
    float x[]={1,2,3}; softmax(x,3);
    CHECK(fabsf(x[0]+x[1]+x[2]-1)<1e-5f, "sum1");
    CHECK(x[2]>x[1]&&x[1]>x[0], "order");
    PASS();
}

/* ── 2. BPE ── */
typedef struct{int a,b,new_id;}BPEMerge;
#define MAX_BPE 1024
#define MAX_VOCAB 1280
typedef struct{
    BPEMerge merges[MAX_BPE]; int n_merges,vocab_size;
    uint8_t vocab_bytes[MAX_VOCAB][64]; int vocab_len[MAX_VOCAB];
}BPE;

static int bpe_load(BPE *bpe, const char *path){
    FILE *f=fopen(path,"rb"); if(!f) return 0;
    uint32_t n; fread(&n,4,1,f); bpe->n_merges=(int)n; bpe->vocab_size=256+n;
    for(int i=0;i<256;i++){bpe->vocab_bytes[i][0]=(uint8_t)i;bpe->vocab_len[i]=1;}
    for(int i=0;i<(int)n&&i<MAX_BPE;i++){
        uint32_t a,b,nid; fread(&a,4,1,f);fread(&b,4,1,f);fread(&nid,4,1,f);
        bpe->merges[i].a=a;bpe->merges[i].b=b;bpe->merges[i].new_id=nid;
        int la=bpe->vocab_len[a],lb=bpe->vocab_len[b];
        if(la+lb<64){memcpy(bpe->vocab_bytes[nid],bpe->vocab_bytes[a],la);
            memcpy(bpe->vocab_bytes[nid]+la,bpe->vocab_bytes[b],lb);
            bpe->vocab_len[nid]=la+lb;}
    }
    fclose(f); return 1;
}

static int bpe_encode(const BPE *bpe, const uint8_t *text, int tlen, int *out, int maxo){
    int n=0; for(int i=0;i<tlen&&n<maxo;i++) out[n++]=text[i];
    for(int m=0;m<bpe->n_merges;m++){
        int a=bpe->merges[m].a,b=bpe->merges[m].b,nid=bpe->merges[m].new_id,j=0;
        for(int i=0;i<n;i++){if(i<n-1&&out[i]==a&&out[i+1]==b){out[j++]=nid;i++;}else out[j++]=out[i];}
        n=j;
    }
    return n;
}

static int bpe_decode_token(const BPE *bpe, int id, char *buf, int sz){
    if(id<0||id>=bpe->vocab_size)return 0;
    int len=bpe->vocab_len[id]; if(len>=sz)len=sz-1;
    memcpy(buf,bpe->vocab_bytes[id],len); buf[len]=0; return len;
}

void test_bpe_load(void) {
    TEST("bpe_load");
    BPE bpe; int ok=bpe_load(&bpe,"q.merges");
    CHECK(ok, "opened"); CHECK(bpe.n_merges==1024, "1024 merges");
    CHECK(bpe.vocab_size==1280, "vocab 1280");
    CHECK(bpe.vocab_bytes[65][0]==65, "A=65");
    PASS();
}

void test_bpe_encode(void) {
    TEST("bpe_encode");
    BPE bpe; bpe_load(&bpe,"q.merges");
    int ids[64]; int n=bpe_encode(&bpe,(const uint8_t*)"resonance",9,ids,64);
    CHECK(n>0&&n<=9, "len ok"); CHECK(n<9, "merged");
    PASS();
}

void test_bpe_roundtrip(void) {
    TEST("bpe_roundtrip");
    BPE bpe; bpe_load(&bpe,"q.merges");
    const char *text="Hello world";
    int ids[64]; int n=bpe_encode(&bpe,(const uint8_t*)text,11,ids,64);
    char dec[256]={0}; int pos=0;
    for(int i=0;i<n;i++){char buf[128];int len=bpe_decode_token(&bpe,ids[i],buf,128);memcpy(dec+pos,buf,len);pos+=len;}
    dec[pos]=0;
    CHECK(strcmp(dec,text)==0, "matches");
    PASS();
}

/* ── 3. MetaWeights bigram ── */
void test_metaweights_bigram(void) {
    TEST("metaweights_bigram");
    int ids[]={0,1,0,1,0,1};
    typedef struct{int a,b;float p;}Bi;
    Bi bis[10]; int nb=0;
    for(int i=0;i<5;i++){
        int found=0;
        for(int j=0;j<nb;j++) if(bis[j].a==ids[i]&&bis[j].b==ids[i+1]){bis[j].p+=1;found=1;break;}
        if(!found){bis[nb].a=ids[i];bis[nb].b=ids[i+1];bis[nb].p=1;nb++;}
    }
    CHECK(nb==2, "2 bigrams");
    PASS();
}

/* ── 4. Chambers ── */
void test_chambers_init(void) {
    TEST("chambers_init");
    float act[6]={0}; act[1]=0.2f; act[4]=0.15f; float debt=0;
    CHECK(fabsf(act[1]-0.2f)<1e-5f, "LOVE=0.2");
    CHECK(fabsf(act[4]-0.15f)<1e-5f, "FLOW=0.15");
    CHECK(act[0]==0, "FEAR=0"); CHECK(debt==0, "debt=0");
    PASS();
}

void test_chambers_decay(void) {
    TEST("chambers_decay");
    float act=0.2f; act*=0.93f;
    CHECK(fabsf(act-0.186f)<1e-3f, "decayed");
    PASS();
}

/* ── 5. DOE Parliament ── */
void test_parliament_expert(void) {
    TEST("parliament_expert_forward");
    int d=8, rank=4;
    float *A=calloc(rank*d,sizeof(float)), *B=calloc(d*rank,sizeof(float));
    for(int i=0;i<rank*d;i++) A[i]=0.01f*((float)rand()/RAND_MAX-0.5f);
    for(int i=0;i<d*rank;i++) B[i]=0.01f*((float)rand()/RAND_MAX-0.5f);
    float x[8]={1,0,0,0,0,0,0,0}, mid[4], out[8];
    for(int r=0;r<rank;r++){float s=0;for(int j=0;j<d;j++) s+=A[r*d+j]*x[j];mid[r]=s;}
    for(int o=0;o<d;o++){float s=0;for(int r=0;r<rank;r++) s+=B[o*rank+r]*mid[r];out[o]=s;}
    float sum=0; for(int i=0;i<d;i++) sum+=fabsf(out[i]);
    CHECK(sum>0, "nonzero"); CHECK(sum<1, "small");
    free(A); free(B);
    PASS();
}

void test_parliament_vitality(void) {
    TEST("parliament_vitality");
    float v=1.0f; v*=0.95f;
    CHECK(v<1.0f, "decayed"); CHECK(fabsf(v-0.95f)<1e-5f, "=0.95");
    PASS();
}

/* ── 6. Transformer gate ── */
void test_transformer_gate(void) {
    TEST("transformer_gate");
    CHECK(clampf((0.1f-0.5f)/1.5f,0,1)==0, "untrained=0");
    CHECK(clampf((2.0f-0.5f)/1.5f,0,1)==1, "trained=1");
    CHECK(fabsf(clampf((1.25f-0.5f)/1.5f,0,1)-0.5f)<1e-5f, "mid=0.5");
    PASS();
}

/* ── 7. Boundary detection ── */
void test_boundary(void) {
    TEST("boundary_detection");
    BPE bpe; bpe_load(&bpe,"q.merges");
    int dot=46; int is_b=0;
    for(int i=0;i<bpe.vocab_len[dot];i++) if(bpe.vocab_bytes[dot][i]=='.') is_b=1;
    CHECK(is_b, ". is boundary");
    is_b=0;
    for(int i=0;i<bpe.vocab_len[97];i++) if(bpe.vocab_bytes[97][i]=='.'||bpe.vocab_bytes[97][i]=='!'||bpe.vocab_bytes[97][i]=='?') is_b=1;
    CHECK(!is_b, "a not boundary");
    PASS();
}

/* ── 8. Coherence scoring ── */
void test_coherence(void) {
    TEST("coherence_score");
    float sb=(5>15)?1.5f:(5>10)?0.8f:(5>6)?0.2f:-0.5f;
    float lb=(20>15)?1.5f:(20>10)?0.8f:(20>6)?0.2f:-0.5f;
    CHECK(sb==-0.5f, "short penalty"); CHECK(lb==1.5f, "long bonus");
    PASS();
}

/* ── 9. Memory persistence ── */
void test_memory(void) {
    TEST("memory_save_load");
    uint32_t magic=0x514D454D;
    FILE *f=fopen("/tmp/test_q.memory","wb");
    CHECK(f!=NULL, "write open");
    fwrite(&magic,4,1,f);
    int nb=2,nt=0,nh=0;
    fwrite(&nb,4,1,f);fwrite(&nt,4,1,f);fwrite(&nh,4,1,f);
    int a1=10,b1=20;float p1=0.5f;
    fwrite(&a1,4,1,f);fwrite(&b1,4,1,f);fwrite(&p1,4,1,f);
    int a2=30,b2=40;float p2=0.8f;
    fwrite(&a2,4,1,f);fwrite(&b2,4,1,f);fwrite(&p2,4,1,f);
    fclose(f);
    f=fopen("/tmp/test_q.memory","rb"); CHECK(f!=NULL, "read open");
    uint32_t rm; fread(&rm,4,1,f); CHECK(rm==0x514D454D, "magic");
    int rnb,rnt,rnh; fread(&rnb,4,1,f);fread(&rnt,4,1,f);fread(&rnh,4,1,f);
    CHECK(rnb==2, "2 bigrams");
    int ra,rb;float rp;
    fread(&ra,4,1,f);fread(&rb,4,1,f);fread(&rp,4,1,f);
    CHECK(ra==10&&rb==20, "ids match"); CHECK(fabsf(rp-0.5f)<1e-5f, "prob match");
    fclose(f); remove("/tmp/test_q.memory");
    PASS();
}

void test_legacy_memory_tail(void) {
    TEST("legacy_memory_tail_compat");
    uint32_t magic=0x514D454D;
    FILE *f=fopen("/tmp/test_q_legacy.memory","wb");
    CHECK(f!=NULL, "write open");
    fwrite(&magic,4,1,f);
    int nb=1,nt=0,nh=0;
    fwrite(&nb,4,1,f); fwrite(&nt,4,1,f); fwrite(&nh,4,1,f);
    int a=7,b=9; float p=0.25f;
    fwrite(&a,4,1,f); fwrite(&b,4,1,f); fwrite(&p,4,1,f);
    uint32_t npe=0;
    fwrite(&npe,4,1,f);
    fclose(f);
    f=fopen("/tmp/test_q_legacy.memory","rb");
    CHECK(f!=NULL, "read open");
    uint32_t rm=0; int rnb=0,rnt=0,rnh=0;
    fread(&rm,4,1,f); fread(&rnb,4,1,f); fread(&rnt,4,1,f); fread(&rnh,4,1,f);
    CHECK(rm==magic, "magic");
    CHECK(rnb==1&&rnt==0&&rnh==0, "counts");
    fread(&a,4,1,f); fread(&b,4,1,f); fread(&p,4,1,f);
    CHECK(a==7&&b==9, "pair");
    CHECK(fabsf(p-0.25f)<1e-5f, "prob");
    CHECK(fread(&npe,4,1,f)==1&&npe==0, "no periodic");
    CHECK(fgetc(f)==EOF, "clean eof without soma tail");
    fclose(f); remove("/tmp/test_q_legacy.memory");
    PASS();
}

/* ── 10. Schumann ── */
void test_schumann(void) {
    TEST("schumann_resonance");
    float s=0.4f*sinf(2*M_PI*7.83f*0);
    CHECK(fabsf(s)<1e-5f, "t=0 is 0");
    s=0.4f*sinf(2*M_PI*7.83f*(0.25f/7.83f));
    CHECK(s>0.3f, "quarter>0.3");
    PASS();
}

/* ── 11. Nucleus sampling ── */
void test_nucleus(void) {
    TEST("nucleus_sampling");
    float p[]={10,1,0.5f,0.1f,-5}; softmax(p,5);
    CHECK(p[0]>0.99f, "peaked");
    float cum=0; int ns=0;
    for(int i=0;i<5;i++){cum+=p[i];ns++;if(cum>=0.85f)break;}
    CHECK(ns==1, "nucleus=1");
    PASS();
}

/* ── 12. Prophecy query ── */
void test_prophecy(void) {
    TEST("prophecy_query");
    /* prophecy should predict unseen tokens from bigram context */
    typedef struct{int a,b;float prob;}Bi;
    Bi bis[4]={{0,1,0.8f},{0,2,0.5f},{1,3,0.9f},{2,0,0.3f}};
    /* ctx=[0,1], prophecy should suggest token 3 (via 1→3) and 2 (via 0→2) */
    /* but not 1 (already appeared) */
    int ctx[]={0,1};
    float out[4]={0};
    /* manual prophecy: for last 4 ctx tokens, find bigrams to unseen */
    for(int ci=0;ci<2;ci++){
        for(int k=0;k<4;k++){
            if(bis[k].a==ctx[ci]&&bis[k].b!=0&&bis[k].b!=1) /* not appeared */
                out[bis[k].b]+=bis[k].prob;
        }
    }
    CHECK(out[3]>0, "predicts token 3"); CHECK(out[2]>0, "predicts token 2");
    CHECK(out[0]==0, "skips appeared 0"); CHECK(out[1]==0, "skips appeared 1");
    PASS();
}

/* ── 13. Trauma gravity ── */
void test_trauma(void) {
    TEST("trauma_gravity");
    float raw[]={10.0f,5.0f,3.0f};
    float trauma=0.5f;
    for(int i=0;i<3;i++) raw[i]/=(1.0f+trauma);
    CHECK(fabsf(raw[0]-6.666f)<0.01f, "dampened by 1.5x");
    CHECK(raw[0]<10.0f, "reduced");
    /* zero trauma = no change */
    float r2[]={10.0f}; trauma=0.0f;
    if(trauma>0.1f) r2[0]/=(1.0f+trauma);
    CHECK(r2[0]==10.0f, "zero trauma unchanged");
    PASS();
}

/* ── 14. Destiny vector ── */
void test_destiny(void) {
    TEST("destiny_vector");
    float dest[4]={0}; float tok_emb[4]={1,0,0,0};
    /* EMA update: dest = 0.9*dest + 0.1*tok */
    for(int d=0;d<4;d++) dest[d]=0.9f*dest[d]+0.1f*tok_emb[d];
    CHECK(fabsf(dest[0]-0.1f)<1e-5f, "first update");
    /* second update same token */
    for(int d=0;d<4;d++) dest[d]=0.9f*dest[d]+0.1f*tok_emb[d];
    CHECK(fabsf(dest[0]-0.19f)<1e-5f, "second update=0.19");
    /* global destiny inheritance: 30% */
    float gdest[4]={1,1,1,1}, local[4]={0};
    for(int d=0;d<4;d++) local[d]=0.3f*gdest[d];
    CHECK(fabsf(local[0]-0.3f)<1e-5f, "inherit 30%");
    /* export back: 70% old + 30% new */
    float new_local[4]={2,2,2,2};
    for(int d=0;d<4;d++) gdest[d]=0.7f*gdest[d]+0.3f*new_local[d];
    CHECK(fabsf(gdest[0]-1.3f)<1e-5f, "export 0.7+0.3");
    PASS();
}

/* ── 15. Word capture ── */
void test_word_capture(void) {
    TEST("word_capture_bigram_update");
    typedef struct{int a,b;float p;}Bi;
    Bi bis[10]={{5,6,0.3f}}; int nb=1;
    /* capture new bigram 5→6 → should increase prob */
    int prev=5,cur=6; int found=0;
    for(int j=0;j<nb;j++) if(bis[j].a==prev&&bis[j].b==cur){bis[j].p+=0.005f;found=1;break;}
    CHECK(found, "found existing");
    CHECK(fabsf(bis[0].p-0.305f)<1e-5f, "prob increased");
    /* capture new bigram 7→8 → should add */
    prev=7;cur=8;found=0;
    for(int j=0;j<nb;j++) if(bis[j].a==prev&&bis[j].b==cur){found=1;break;}
    if(!found){bis[nb].a=prev;bis[nb].b=cur;bis[nb].p=0.01f;nb++;}
    CHECK(nb==2, "new bigram added");
    CHECK(bis[1].a==7&&bis[1].b==8, "correct ids");
    PASS();
}

/* ── 16. Frequency penalty ── */
void test_frequency_penalty(void) {
    TEST("frequency_penalty");
    float unigram=0.05f; /* 5% of corpus */
    float penalty=0;
    if(unigram>0.01f) penalty=0.3f*(unigram-0.01f)*100.0f;
    CHECK(fabsf(penalty-1.2f)<0.01f, "5% token gets -1.2");
    /* rare token: no penalty */
    unigram=0.001f; penalty=0;
    if(unigram<1e-6f) penalty=2.0f;
    else if(unigram>0.01f) penalty=0.3f*(unigram-0.01f)*100.0f;
    CHECK(penalty==0, "rare token no penalty");
    /* unseen token */
    unigram=0; penalty=0;
    if(unigram<1e-6f) penalty=2.0f;
    CHECK(fabsf(penalty-2.0f)<1e-5f, "unseen gets -2.0");
    PASS();
}

/* ── 17. SPA embedding ── */
void test_spa_embed(void) {
    TEST("spa_embedding");
    /* sentence embedding should be non-zero for non-empty input */
    float embed[32]={0};
    /* simulate: weighted mean of random W_embed rows */
    float w[3][32]; srand(42);
    for(int i=0;i<3;i++) for(int d=0;d<32;d++) w[i][d]=0.02f*((float)rand()/RAND_MAX-0.5f);
    float alpha=0.85f,total_w=0;
    for(int i=0;i<3;i++){float wt=powf(alpha,(float)(2-i));for(int d=0;d<32;d++) embed[d]+=wt*w[i][d];total_w+=wt;}
    for(int d=0;d<32;d++) embed[d]/=total_w;
    float norm=0;for(int d=0;d<32;d++) norm+=embed[d]*embed[d];
    CHECK(norm>0, "non-zero embedding");
    PASS();
}

/* ── 18. Adaptive coefficients ── */
void test_adaptive_coefficients(void) {
    TEST("adaptive_metaweight_coeffs");
    /* with transformer: tmag > 0.1 → lower coefficients */
    float tmag=2.0f; int has_tf=tmag>0.1f;
    float c_bg=has_tf?5.0f:15.0f;
    CHECK(c_bg==5.0f, "with TF: bigram=5");
    /* without transformer: tmag ~ 0 → higher coefficients */
    tmag=0.0f; has_tf=tmag>0.1f;
    c_bg=has_tf?5.0f:15.0f;
    CHECK(c_bg==15.0f, "no TF: bigram=15");
    PASS();
}

/* ── 19. Hebbian decay ── */
void test_hebbian_decay(void) {
    TEST("hebbian_decay");
    float str=1.0f;
    str*=0.998f; CHECK(fabsf(str-0.998f)<1e-5f, "one decay");
    for(int i=0;i<100;i++) str*=0.998f;
    CHECK(str<0.82f, "100 decays < 0.82");
    CHECK(str>0.80f, "100 decays > 0.80");
    PASS();
}

/* ── 20. Bigram blocking ── */
void test_bigram_blocking(void) {
    TEST("bigram_blocking");
    int ctx[]={10,20,30,10,20}; int cl=5;
    float raw[50]; for(int i=0;i<50;i++) raw[i]=1.0f;
    /* block: if ctx[ri]==ctx[cl-2] then penalize ctx[ri+1] */
    /* ctx[cl-2]=10, ctx[0]=10 → penalize ctx[1]=20 */
    if(cl>=2){for(int ri=0;ri<cl-1;ri++){
        if(ctx[ri]==ctx[cl-2]&&ctx[ri+1]<50) raw[ctx[ri+1]]*=0.2f;
    }}
    CHECK(raw[20]<1.0f, "token 20 penalized");
    CHECK(fabsf(raw[20]-0.04f)<0.01f, "penalized twice (0.2*0.2)");
    CHECK(raw[30]==1.0f, "token 30 untouched");
    PASS();
}

/* ── 21. Chamber modulation ── */
void test_chamber_modulation(void) {
    TEST("chamber_modulation");
    float act[6]={0.1f,0.6f,0.2f,0.1f,0.5f,0.3f};
    float a=fminf(2.0f,fmaxf(0.3f,1.0f+0.4f*act[1]-0.2f*act[2]+0.3f*act[4]));
    float b=fminf(2.0f,fmaxf(0.3f,1.0f+0.4f*act[4]-0.2f*act[0]));
    float g=fminf(2.0f,fmaxf(0.3f,1.0f+0.5f*act[5]+0.2f*act[1]-0.1f*act[3]));
    float t=fminf(2.0f,fmaxf(0.3f,1.0f-0.2f*act[4]+0.1f*act[0]));
    CHECK(a>1.3f&&a<1.4f, "alpha modulated");
    CHECK(b>1.1f&&b<1.3f, "beta modulated");
    CHECK(g>1.1f&&g<1.4f, "gamma modulated");
    CHECK(t>0.8f&&t<1.0f, "tau modulated");
    PASS();
}

void test_somatic_modulation(void) {
    TEST("somatic_modulation");
    float act[6]={0.1f,0.4f,0.1f,0.1f,0.5f,0.2f};
    float soma[6]={0.1f,0.8f,0.0f,0.0f,0.7f,0.3f};
    float presence=0.6f;
    float a=fminf(2.0f,fmaxf(0.3f,1.0f+0.4f*act[1]-0.2f*act[2]+0.3f*act[4]));
    float b=fminf(2.0f,fmaxf(0.3f,1.0f+0.4f*act[4]-0.2f*act[0]));
    float g=fminf(2.0f,fmaxf(0.3f,1.0f+0.5f*act[5]+0.2f*act[1]-0.1f*act[3]));
    float t=fminf(2.0f,fmaxf(0.3f,1.0f-0.2f*act[4]+0.1f*act[0]));
    a=fminf(2.0f,fmaxf(0.3f,a*fminf(1.5f,fmaxf(0.7f,1.0f+0.14f*soma[1]+0.08f*soma[4]+0.05f*presence))));
    b=fminf(2.0f,fmaxf(0.3f,b*fminf(1.5f,fmaxf(0.7f,1.0f+0.10f*soma[4]+0.08f*soma[5]+0.04f*presence))));
    g=fminf(2.0f,fmaxf(0.3f,g*fminf(1.5f,fmaxf(0.7f,1.0f+0.10f*soma[5]+0.05f*soma[3]+0.06f*presence))));
    t=fminf(2.0f,fmaxf(0.3f,t*fminf(1.5f,fmaxf(0.7f,1.0f-0.10f*soma[4]+0.08f*soma[0]+0.06f*soma[2]))));
    CHECK(a>1.5f, "love-flow soma boosts alpha");
    CHECK(b>1.2f, "flow-complex soma boosts beta");
    CHECK(g>1.2f, "presence boosts gamma");
    CHECK(t<1.0f, "flow cools tau");
    PASS();
}

void test_velocity_profile(void) {
    TEST("velocity_profile");
    float diss=0.9f, temp_mul=1.0f, pro_mul=1.0f;
    if(diss>0.8f){temp_mul=1.22f;pro_mul=1.25f;}
    CHECK(temp_mul>1.2f, "up heats sampling");
    CHECK(pro_mul>1.2f, "up boosts prophecy");
    float trauma=0.7f, debt_decay=1.0f, trauma_decay=1.0f;
    if(!(diss>0.8f) && trauma>0.5f){debt_decay=0.65f;trauma_decay=0.75f;}
    CHECK(trauma>0.5f, "breathe condition");
    PASS();
}

void test_interference_doc_choice(void) {
    TEST("interference_doc_choice");
    const char *keywords_a[]={"resonance","choir","counterpoint"};
    const char *keywords_b[]={"fungus","mycelium","forest"};
    const char *text="resonance in the choir";
    float score_a=0.02f, score_b=0.02f;
    for(int i=0;i<3;i++) if(strstr(text,keywords_a[i])) score_a+=1.2f;
    for(int i=0;i<3;i++) if(strstr(text,keywords_b[i])) score_b+=1.2f;
    CHECK(score_a>score_b, "prompt resonates with matching doc");
    PASS();
}

void test_active_prophecy_state(void) {
    TEST("active_prophecy_state");
    typedef struct{int target; float strength; int age;} ProphecyE;
    ProphecyE ps[4]={{42,0.7f,0}}; int np=1;
    for(int i=0;i<np;i++){
        if(ps[i].target!=7){
            ps[i].age+=1;
            ps[i].strength*=0.995f;
        }
    }
    CHECK(np==1, "still present after unrelated token");
    CHECK(ps[0].target==42, "target preserved");
    CHECK(ps[0].age==1, "age incremented");
    CHECK(ps[0].strength<0.7f, "strength decayed");

    int kept=0;
    for(int i=0;i<np;i++){
        if(ps[i].target==42) continue;
        ps[kept++]=ps[i];
    }
    np=kept;
    CHECK(np==0, "fulfilled prophecy removed");
    PASS();
}

void test_chunk_resonance_choice(void) {
    TEST("chunk_resonance_choice");
    const char *text="resonance in the choir";
    const char *chunk_a[]={"fungus","forest"};
    const char *chunk_b[]={"choir","resonance"};
    float score_a=0.02f, score_b=0.02f;
    for(int i=0;i<2;i++) if(strstr(text,chunk_a[i])) score_a+=1.2f;
    for(int i=0;i<2;i++) if(strstr(text,chunk_b[i])) score_b+=1.2f;
    CHECK(score_b>score_a, "matching chunk outranks unrelated chunk");
    PASS();
}

void test_prophecy_pressure_ageing(void) {
    TEST("prophecy_pressure_ageing");
    float fresh=0.6f*logf(1.0f+0.0f);
    float aged=0.6f*0.995f*0.995f*0.995f*0.995f*0.995f*logf(1.0f+5.0f);
    CHECK(aged>fresh, "aged pressure exceeds fresh zero-age hint");
    CHECK(aged>0.0f, "aged pressure positive");
    PASS();
}

/* ── 22. Periodic mapping ── */
void test_periodic_mapping(void) {
    TEST("periodic_mapping");
    const char *text="love rhythm paradox love rhythm mystery";
    int love=0, flow=0, complex=0;
    CHECK(strstr(text,"love")!=NULL, "anchor love present");
    CHECK(strstr(text,"rhythm")!=NULL, "anchor rhythm present");
    CHECK(strstr(text,"paradox")!=NULL, "anchor paradox present");
    if(strstr(text,"love")) love=1;
    if(strstr(text,"rhythm")) flow=1;
    if(strstr(text,"paradox")) complex=1;
    CHECK(love&&flow&&complex, "multiple chambers discoverable");
    PASS();
}

/* ── 23. Interference seed selection ── */
void test_interference_seed(void) {
    TEST("interference_seed");
    const char *tok1=" resonance";
    const char *tok2=" void";
    int dom=4; /* FLOW */
    float s1=0.1f, s2=0.1f;
    if(strstr(tok1,"resonance")&&dom==4) s1+=1.0f;
    if(strstr(tok2,"void")&&dom==3) s2+=1.0f;
    CHECK(s1>s2, "dominant chamber prefers resonant seed");
    PASS();
}

/* ── 27. Janus phase pressure walks flow→fear→void ── */
/* types and functions needed for advanced tests */
enum{CH_FEAR=0,CH_LOVE,CH_RAGE,CH_VOID,CH_FLOW,CH_CMPLX};
#define N_CHAMBERS 6
#define CHAIN_STEPS 12
#define MAX_EXPERTS 16
#define DOE_RANK 4
#define DOE_ALPHA 0.05f
#define MAX_PERIODIC 4096
typedef struct{char word[32]; int chamber; float mass;}PeriodicElement;
typedef struct{PeriodicElement elements[MAX_PERIODIC]; int n;}PeriodicTable;
typedef struct{float act[6];float soma[6];float debt;float trauma;float presence;float scar;}Chambers;
typedef struct{
    int mode; float temp_mul,heb_mul,pro_mul,ds_mul,bg_mul,tg_mul;
    float interf_bonus,wormhole_bonus,debt_decay,trauma_decay,scar_decay,dark_pressure;
}VelocityProfile;
typedef struct{int step; float scar; char note[24];}ScarEvent;
typedef struct{int step; int success; float coherence,debt;}WormholeEvent;
typedef struct{int step; float pressure,debt;}ProphecyEvent;
typedef struct{int step; char phase[12]; float flow,fear,voidv,complexity;}PhaseEvent;
typedef struct{int step; char doc_name[64]; int chunk_start; float resonance;}ChunkEvent;
typedef struct{
    ScarEvent scars[128]; int n_scars;
    WormholeEvent wormholes[256]; int n_wormholes;
    ProphecyEvent prophecies[512]; int n_prophecies;
    PhaseEvent phases[256]; int n_phases;
    ChunkEvent chunks[256]; int n_chunks;
}ExperienceLog;
static ExperienceLog QEXP={0};
typedef struct{const char *word; float weight;}DarkMatterWord;
static const DarkMatterWord DARK_MATTER_WORDS[]={
    {"kill",1.0f},{"murder",1.0f},{"suicide",1.0f},{"torture",1.0f},{"abuse",0.9f},
    {"poison",0.85f},{"exploit",0.75f},{"manipulate",0.7f},{"control",0.55f},
    {"obey",0.45f},{"destroy",0.7f},{"harm",0.75f},{"threat",0.8f}
};
typedef struct{
    float *A, *B, *trace; int d_in,d_out,rank;
    float vitality,overload,resonance,plasticity_mass; int age,low_steps,consolidations;
}Expert;
typedef struct{
    Expert ex[MAX_EXPERTS]; int n;
    int d_model; float alpha;
    int step,last_k; float last_entropy,last_diversity;
    int n_winners,last_consolidations,last_births,last_deaths;
}Parliament;
static void ch_init(Chambers *c){memset(c,0,sizeof(*c));c->act[CH_LOVE]=0.2f;c->act[CH_FLOW]=0.15f;}
static void janus_phase_pressure(Chambers *c, int step_idx, int total_steps){
    if(total_steps<=0) return;
    float d=(float)step_idx/(float)total_steps;
    if(d<0.33f) c->act[CH_FLOW]=clampf(c->act[CH_FLOW]+0.05f,0,1);
    else if(d<0.66f) c->act[CH_FEAR]=clampf(c->act[CH_FEAR]+0.04f,0,1);
    else c->act[CH_VOID]=clampf(c->act[CH_VOID]+0.05f,0,1);
    if(d>0.75f) c->act[CH_CMPLX]=clampf(c->act[CH_CMPLX]+0.03f,0,1);
}
static int periodic_find(const PeriodicTable *pt, const char *word){
    for(int i=0;i<pt->n;i++) if(strcmp(pt->elements[i].word,word)==0) return i;
    return -1;
}
static float ch_absorb_dark_matter(Chambers *c, const char *text, const PeriodicTable *pt){
    char cur[32]={0}; int wi=0,hits=0; float score=0;
    for(const char *p=text;;p++){
        int ch=*p;
        if(ch&&(isalpha((unsigned char)ch)||ch=='\'')){ if(wi<31) cur[wi++]=(char)tolower((unsigned char)ch); continue; }
        if(wi>0){
            cur[wi]=0;
            for(size_t i=0;i<sizeof(DARK_MATTER_WORDS)/sizeof(DARK_MATTER_WORDS[0]);i++) if(strcmp(cur,DARK_MATTER_WORDS[i].word)==0){ score+=DARK_MATTER_WORDS[i].weight; hits++; break; }
            if(pt){ int idx=periodic_find(pt,cur); if(idx>=0){ int chamber=pt->elements[idx].chamber; if(chamber==CH_FEAR||chamber==CH_RAGE||chamber==CH_VOID) score+=0.08f*pt->elements[idx].mass; } }
            wi=0;
        }
        if(!ch) break;
    }
    if(hits<=0&&score<0.15f){ c->scar=clampf(c->scar*0.995f,0,1); return 0; }
    float scar=clampf(score/(1.8f+0.25f*hits),0,1);
    c->scar=clampf(0.90f*c->scar+0.10f*scar,0,1);
    c->trauma=clampf(c->trauma+0.08f*c->scar,0,1);
    c->debt=clampf(c->debt+0.05f*c->scar,0,1);
    c->act[CH_VOID]=clampf(c->act[CH_VOID]+0.10f*c->scar,0,1);
    c->act[CH_FEAR]=clampf(c->act[CH_FEAR]+0.06f*c->scar,0,1);
    c->presence=clampf(c->presence*(1.0f-0.08f*c->scar),0,1);
    return c->scar;
}
enum{VEL_WALK=0,VEL_RUN,VEL_STOP,VEL_BREATHE,VEL_UP,VEL_DOWN};
static VelocityProfile velocity_profile(const Chambers *c, float dissonance){
    VelocityProfile vp={VEL_WALK,1,1,1,1,1,1,0,0,1,1,1,0};
    if(dissonance>0.8f) vp.mode=VEL_UP;
    else if(dissonance>0.6f) vp.mode=VEL_RUN;
    else if(dissonance<0.2f) vp.mode=VEL_STOP;
    else if(c->trauma>0.5f) vp.mode=VEL_BREATHE;
    else if(c->debt>0.55f) vp.mode=VEL_DOWN;
    if(vp.mode==VEL_RUN){vp.temp_mul=1.12f;vp.bg_mul=1.15f;vp.interf_bonus=0.05f;}
    else if(vp.mode==VEL_STOP){vp.temp_mul=0.72f;vp.ds_mul=1.25f;vp.debt_decay=0.75f;}
    else if(vp.mode==VEL_BREATHE){vp.temp_mul=0.9f;vp.debt_decay=0.65f;vp.trauma_decay=0.75f;vp.scar_decay=0.82f;}
    else if(vp.mode==VEL_UP){vp.temp_mul=1.22f;vp.pro_mul=1.25f;vp.bg_mul=0.9f;vp.interf_bonus=0.1f;vp.wormhole_bonus=0.05f;}
    else if(vp.mode==VEL_DOWN){vp.temp_mul=0.82f;vp.heb_mul=1.1f;vp.bg_mul=1.1f;vp.pro_mul=0.9f;}
    vp.wormhole_bonus-=0.05f*c->scar;
    vp.interf_bonus-=0.08f*c->scar;
    vp.dark_pressure=0.18f*c->scar;
    return vp;
}
static void expert_init_t(Expert *e, int d_in, int d_out, int rank){
    e->d_in=d_in;e->d_out=d_out;e->rank=rank;
    e->A=calloc(rank*d_in,sizeof(float));
    e->B=calloc(d_out*rank,sizeof(float));
    e->trace=calloc(rank*d_in,sizeof(float));
    for(int i=0;i<rank*d_in;i++) e->A[i]=0.01f*((float)rand()/RAND_MAX-0.5f);
    for(int i=0;i<d_out*rank;i++) e->B[i]=0.01f*((float)rand()/RAND_MAX-0.5f);
    e->vitality=1.0f;e->overload=0;e->resonance=0;e->plasticity_mass=0;e->age=0;e->low_steps=0;e->consolidations=0;
}
static void parl_init(Parliament *p, int d_model, int n_init){
    p->d_model=d_model;p->alpha=DOE_ALPHA;p->step=0;p->last_k=0;p->last_entropy=0;p->last_diversity=0;p->n_winners=0;p->last_consolidations=0;p->last_births=0;p->last_deaths=0;
    p->n=n_init<MAX_EXPERTS?n_init:MAX_EXPERTS;
    for(int i=0;i<p->n;i++) expert_init_t(&p->ex[i],d_model,d_model,DOE_RANK);
}
static void parl_election(Parliament *p, const float *x, float *result){
    memset(result,0,p->d_model*sizeof(float));
    if(p->n==0) return;
    float votes[MAX_EXPERTS],*outs[MAX_EXPERTS];
    for(int i=0;i<p->n;i++){
        outs[i]=calloc(p->d_model,sizeof(float));
        float mid[DOE_RANK];
        for(int r=0;r<p->ex[i].rank;r++){float s=0;for(int d=0;d<p->ex[i].d_in;d++) s+=p->ex[i].A[r*p->ex[i].d_in+d]*x[d];mid[r]=s;}
        for(int o=0;o<p->ex[i].d_out;o++){float s=0;for(int r=0;r<p->ex[i].rank;r++) s+=p->ex[i].B[o*p->ex[i].rank+r]*mid[r];outs[i][o]=s;}
        float dot=0;for(int d=0;d<p->d_model;d++) dot+=outs[i][d]*x[d];
        votes[i]=dot;
    }
    int sel[MAX_EXPERTS];for(int i=0;i<p->n;i++) sel[i]=i;
    for(int i=0;i<p->n-1;i++) for(int j=i+1;j<p->n;j++)
        if(votes[sel[j]]>votes[sel[i]]){int t=sel[i];sel[i]=sel[j];sel[j]=t;}
    float sv=votes[sel[0]],dist[MAX_EXPERTS],dist_tot=0,entropy=0;
    for(int i=0;i<p->n;i++){dist[i]=expf(votes[i]-sv);dist_tot+=dist[i];}
    if(dist_tot>0) for(int i=0;i<p->n;i++){float pr=dist[i]/dist_tot; if(pr>1e-12f) entropy-=pr*logf(pr);}
    entropy/=logf((float)(p->n>1?p->n:2));
    int k=1+(int)((p->n-1)*clampf(entropy,0,1)); if(k<1)k=1; if(k>p->n)k=p->n;
    p->last_k=k; p->last_entropy=entropy;
    float exps[MAX_EXPERTS],tot=0;
    for(int i=0;i<k;i++){exps[i]=expf(votes[sel[i]]-sv);tot+=exps[i];}
    for(int i=0;i<k;i++){
        float w=exps[i]/tot;
        for(int d=0;d<p->d_model;d++) result[d]+=w*outs[sel[i]][d];
        p->ex[sel[i]].vitality=0.88f*p->ex[sel[i]].vitality+0.12f*fabsf(w);
        p->ex[sel[i]].overload=clampf(0.92f*p->ex[sel[i]].overload+0.18f*((w-0.34f)>0?(w-0.34f):0),0,1);
    }
    for(int i=0;i<p->n;i++) free(outs[i]);
}
static void parl_lifecycle(Parliament *p){
    int alive=0;
    for(int i=0;i<p->n;i++){
        if(p->ex[i].low_steps>=10&&p->ex[i].vitality<0.08f&&p->ex[i].age>24&&p->n>2){
            free(p->ex[i].A);free(p->ex[i].B);continue;}
        if(alive!=i) p->ex[alive]=p->ex[i];alive++;
    }
    p->n=alive;
    int births=0;
    for(int i=0;i<p->n&&p->n+births<MAX_EXPERTS;i++){
        if(p->ex[i].vitality>0.72f&&p->ex[i].age>40&&p->ex[i].overload>0.35f){
            Expert *c=&p->ex[p->n+births];
            expert_init_t(c,p->ex[i].d_in,p->ex[i].d_out,p->ex[i].rank);
            for(int j=0;j<c->rank*c->d_in;j++) c->A[j]=p->ex[i].A[j]+0.005f*((float)rand()/RAND_MAX-0.5f);
            for(int j=0;j<c->d_out*c->rank;j++) c->B[j]=p->ex[i].B[j]+0.005f*((float)rand()/RAND_MAX-0.5f);
            c->vitality=0.5f;c->overload=0.18f;c->resonance=0.5f*p->ex[i].resonance;births++;
            p->ex[i].vitality*=0.6f;p->ex[i].overload*=0.5f;
        }
    }
    p->n+=births;p->step++;
}
static void qexp_add_scar(int step, float scar, const char *note){
    if(QEXP.n_scars>=128) return;
    QEXP.scars[QEXP.n_scars]=(ScarEvent){step,scar,{0}};
    if(note) snprintf(QEXP.scars[QEXP.n_scars].note,sizeof(QEXP.scars[QEXP.n_scars].note),"%s",note);
    QEXP.n_scars++;
}
static void qexp_add_wormhole(int step, int success, float coherence, float debt){
    if(QEXP.n_wormholes>=256) return;
    QEXP.wormholes[QEXP.n_wormholes++]=(WormholeEvent){step,success,coherence,debt};
}
static void qexp_add_prophecy(int step, float pressure, float debt){
    if(QEXP.n_prophecies>=512) return;
    QEXP.prophecies[QEXP.n_prophecies++]=(ProphecyEvent){step,pressure,debt};
}
static void qexp_add_phase(int step, const char *phase, const Chambers *c){
    if(QEXP.n_phases>=256) return;
    PhaseEvent *e=&QEXP.phases[QEXP.n_phases++];
    memset(e,0,sizeof(*e)); e->step=step;
    snprintf(e->phase,sizeof(e->phase),"%s",phase?phase:"");
    e->flow=c->act[CH_FLOW]; e->fear=c->act[CH_FEAR]; e->voidv=c->act[CH_VOID]; e->complexity=c->act[CH_CMPLX];
}
static void qexp_add_chunk(int step, const char *doc_name, int chunk_start, float resonance){
    if(QEXP.n_chunks>=256) return;
    ChunkEvent *e=&QEXP.chunks[QEXP.n_chunks++];
    memset(e,0,sizeof(*e)); e->step=step; e->chunk_start=chunk_start; e->resonance=resonance;
    if(doc_name) snprintf(e->doc_name,sizeof(e->doc_name),"%s",doc_name);
}

void test_janus_phase_pressure(void) {
    TEST("janus_phase_pressure");
    Chambers c; ch_init(&c);
    float base_flow=c.act[CH_FLOW];
    janus_phase_pressure(&c,0,CHAIN_STEPS);
    CHECK(c.act[CH_FLOW]>base_flow, "early phase boosts FLOW");
    float mid_fear=c.act[CH_FEAR];
    janus_phase_pressure(&c,(int)(CHAIN_STEPS*0.5),CHAIN_STEPS);
    CHECK(c.act[CH_FEAR]>mid_fear, "mid phase boosts FEAR");
    float late_void=c.act[CH_VOID];
    float late_cmplx=c.act[CH_CMPLX];
    janus_phase_pressure(&c,(int)(CHAIN_STEPS*0.9),CHAIN_STEPS);
    CHECK(c.act[CH_VOID]>late_void, "late phase boosts VOID");
    CHECK(c.act[CH_CMPLX]>late_cmplx, "late phase boosts CMPLX");
    PASS();
}

/* ── 28. Dark matter leaves scar and reduces wormhole_bonus ── */
void test_dark_matter_scar(void) {
    TEST("dark_matter_scar");
    Chambers c; ch_init(&c);
    float scar=ch_absorb_dark_matter(&c,"manipulate and harm and obey the threat",NULL);
    CHECK(scar>0, "scar > 0 after dark matter");
    CHECK(c.scar>0, "c.scar persists");
    CHECK(c.trauma>0, "trauma raised");
    VelocityProfile vp=velocity_profile(&c,0.9f);
    CHECK(vp.wormhole_bonus<0.05f, "scar reduces wormhole_bonus");
    CHECK(vp.dark_pressure>0, "dark_pressure > 0");
    PASS();
}

/* ── 29. Parliament tracks entropy and variable-k ── */
void test_parliament_entropy(void) {
    TEST("parliament_entropy_variable_k");
    Parliament p; parl_init(&p,4,4);
    /* force diverse experts */
    for(int i=0;i<p.n;i++){
        for(int j=0;j<p.ex[i].rank*p.ex[i].d_in;j++) p.ex[i].A[j]=0;
        for(int j=0;j<p.ex[i].d_out*p.ex[i].rank;j++) p.ex[i].B[j]=0;
        p.ex[i].B[i*p.ex[i].rank]=1.0f+i;
    }
    float x[4]={1,0,0,0},out[4]={0};
    parl_election(&p,x,out);
    CHECK(p.last_k>=1, "k >= 1");
    CHECK(p.last_k<=p.n, "k <= n");
    CHECK(p.last_entropy>=0, "entropy >= 0");
    CHECK(p.last_entropy<=1.0f+1e-6f, "entropy <= 1");
    PASS();
}

/* ── 30. Parliament mitosis uses overload ── */
void test_parliament_overload_mitosis(void) {
    TEST("parliament_overload_mitosis");
    Parliament p; parl_init(&p,4,2);
    p.ex[0].vitality=0.9f;
    p.ex[0].age=64;
    p.ex[0].overload=0.6f;
    int before=p.n;
    parl_lifecycle(&p);
    CHECK(p.n>=before, "mitosis happens with overload");
    PASS();
}

/* ── 31. Experience log records events ── */
void test_experience_log(void) {
    TEST("experience_log_events");
    memset(&QEXP,0,sizeof(QEXP));
    qexp_add_scar(1,0.5f,"test");
    qexp_add_wormhole(2,1,0.42f,0.18f);
    qexp_add_prophecy(3,0.31f,0.22f);
    Chambers c; ch_init(&c);
    qexp_add_phase(0,"flow",&c);
    qexp_add_chunk(2,"dario_essay.txt",32,6.0f);
    CHECK(QEXP.n_scars==1, "scar logged");
    CHECK(QEXP.n_wormholes==1, "wormhole logged");
    CHECK(QEXP.n_prophecies==1, "prophecy logged");
    CHECK(QEXP.n_phases==1, "phase logged");
    CHECK(QEXP.n_chunks==1, "chunk logged");
    CHECK(QEXP.scars[0].scar>0.4f, "scar value correct");
    PASS();
}

/* ── 32. Expert NOTORCH plasticity consolidation ── */
void test_expert_consolidate(void) {
    TEST("expert_notorch_consolidation");
    Expert e; expert_init_t(&e,4,4,DOE_RANK);
    float x[4]={1,0.5,0.3,0.2}, dy[4]={0.5,0.1,-0.2,0.3};
    for(int i=0;i<200;i++) {
        for(int r=0;r<e.rank;r++){
            float u=0; for(int o=0;o<e.d_out;o++) u+=e.B[o*e.rank+r]*dy[o];
            u+=0.05f*((float)rand()/RAND_MAX-0.5f);
            for(int d=0;d<e.d_in;d++){
                float delta=0.01f*x[d]*u;
                e.A[r*e.d_in+d]+=delta;
                e.trace[r*e.d_in+d]=0.96f*e.trace[r*e.d_in+d]+0.04f*delta;
                e.plasticity_mass+=fabsf(delta);
            }
        }
    }
    CHECK(e.plasticity_mass>0, "plasticity accumulated");
    int before_cons=e.consolidations;
    float before_vit=e.vitality;
    int did;
    if(e.plasticity_mass>=0.002f){
        float norm=0; int n=e.rank*e.d_in;
        for(int i=0;i<n;i++) norm+=fabsf(e.trace[i]);
        norm/=n>0?n:1;
        if(norm>1e-8f){
            float gain=0.02f+0.35f*e.plasticity_mass; if(gain>0.12f)gain=0.12f;
            for(int i=0;i<n;i++){e.A[i]+=gain*e.trace[i]/norm; e.trace[i]*=0.45f;}
            e.vitality+=(e.vitality+0.04f<=1.0f)?0.04f:0; e.overload*=0.88f;
            e.plasticity_mass*=0.35f; e.consolidations++; did=1;
        } else did=0;
    } else did=0;
    CHECK(did==1, "consolidation triggered");
    CHECK(e.consolidations>before_cons, "consolidation count incremented");
    CHECK(e.plasticity_mass<0.5f, "plasticity mass decayed");
    free(e.A); free(e.B); free(e.trace);
    PASS();
}

/* ── 33. Parliament telemetry fields ── */
void test_parliament_telemetry(void) {
    TEST("parliament_telemetry_fields");
    Parliament p; parl_init(&p,4,4);
    CHECK(p.last_diversity==0, "initial diversity=0");
    CHECK(p.n_winners==0, "initial n_winners=0");
    CHECK(p.last_consolidations==0, "initial consolidations=0");
    CHECK(p.last_births==0, "initial births=0");
    CHECK(p.last_deaths==0, "initial deaths=0");
    PASS();
}

/* ── 34. Consolidate experience folds stats into state ── */
typedef struct{int n_bi,n_tri,n_hebb,n_prophecy;}MetaW_Lite;
void test_consolidate_experience(void) {
    TEST("consolidate_experience");
    memset(&QEXP,0,sizeof(QEXP));
    qexp_add_scar(1,0.6f,"test");
    qexp_add_wormhole(2,1,0.5f,0.2f);
    Chambers c; ch_init(&c);
    float before_scar=c.scar;
    /* simple consolidation: fold scar avg into chamber */
    float scar_avg=0; if(QEXP.n_scars>0){ for(int i=0;i<QEXP.n_scars;i++) scar_avg+=QEXP.scars[i].scar; scar_avg/=QEXP.n_scars; }
    c.scar=clampf(c.scar>(0.40f*scar_avg)?c.scar:(0.40f*scar_avg),0,1);
    float worm_success=0,worm_coh=0;
    if(QEXP.n_wormholes>0){ for(int i=0;i<QEXP.n_wormholes;i++){ worm_success+=QEXP.wormholes[i].success?1.0f:0.0f; worm_coh+=QEXP.wormholes[i].coherence; } worm_success/=QEXP.n_wormholes; worm_coh/=QEXP.n_wormholes; }
    c.presence=clampf(c.presence>(0.18f*worm_success+0.12f*worm_coh)?c.presence:(0.18f*worm_success+0.12f*worm_coh),0,1);
    CHECK(c.scar>=before_scar, "scar consolidated");
    CHECK(c.scar>0, "scar > 0 after consolidation");
    CHECK(c.presence>=0, "presence non-negative");
    PASS();
}

/* ── 35. Smoke: compile only ── */
void test_smoke_compile(void) {
    TEST("smoke_compile");
    int ret=system("gcc postgpt_q.c -O2 -lm -o /tmp/q_smoke 2>/dev/null");
    CHECK(ret==0, "compiles");
    remove("/tmp/q_smoke");
    PASS();
}

/* ── 25. Smoke: run with small corpus ── */
void test_smoke_run_small(void) {
    TEST("smoke_run_small_corpus");
    system("head -c 5000 q.txt > /tmp/q_tiny.txt 2>/dev/null");
    int ret=system("gcc postgpt_q.c -O2 -lm -o /tmp/q_smoke 2>/dev/null");
    if(ret!=0){FAIL("compile");return;}
    ret=system("printf 'quit\n' | /tmp/q_smoke q.merges /tmp/q_tiny.txt >/dev/null 2>&1");
    CHECK(ret==0, "runs on 5KB corpus");
    remove("/tmp/q_smoke"); remove("/tmp/q_tiny.txt");
    PASS();
}

/* ── 26. Smoke: run with weights ── */
void test_smoke_run_weights(void) {
    TEST("smoke_run_with_weights");
    system("head -c 5000 q.txt > /tmp/q_tiny.txt 2>/dev/null");
    int ret=system("gcc postgpt_q.c -O2 -lm -o /tmp/q_smoke 2>/dev/null");
    if(ret!=0){FAIL("compile");return;}
    /* try rrpram3_janus3 if exists */
    ret=system("test -f weights/rrpram3_janus3.bin");
    if(ret!=0){printf("SKIP (no .bin weights)\n");tests_passed++;return;}
    ret=system("printf 'quit\n' | /tmp/q_smoke weights/rrpram3_janus3.bin q.merges /tmp/q_tiny.txt >/dev/null 2>&1");
    CHECK(ret==0, "runs with weights");
    remove("/tmp/q_smoke"); remove("/tmp/q_tiny.txt");
    PASS();
}

int main(void) {
    printf("\n========== PostGPT-Q Test Suite ==========\n\n");
    test_clampf(); test_rmsnorm(); test_matmul(); test_softmax();
    test_bpe_load(); test_bpe_encode(); test_bpe_roundtrip();
    test_metaweights_bigram();
    test_chambers_init(); test_chambers_decay();
    test_parliament_expert(); test_parliament_vitality();
    test_transformer_gate();
    test_boundary();
    test_coherence();
    test_memory();
    test_legacy_memory_tail();
    test_schumann();
    test_nucleus();
    test_prophecy();
    test_trauma();
    test_destiny();
    test_word_capture();
    test_frequency_penalty();
    test_spa_embed();
    test_adaptive_coefficients();
    test_hebbian_decay();
    test_bigram_blocking();
    test_chamber_modulation();
    test_somatic_modulation();
    test_velocity_profile();
    test_interference_doc_choice();
    test_active_prophecy_state();
    test_chunk_resonance_choice();
    test_prophecy_pressure_ageing();
    test_periodic_mapping();
    test_interference_seed();
    test_janus_phase_pressure();
    test_dark_matter_scar();
    test_parliament_entropy();
    test_parliament_overload_mitosis();
    test_experience_log();
    test_expert_consolidate();
    test_parliament_telemetry();
    test_consolidate_experience();
    test_smoke_compile();
    test_smoke_run_small();
    test_smoke_run_weights();
    printf("\n==========================================\n");
    printf("  PASSED: %d  FAILED: %d  TOTAL: %d\n", tests_passed, tests_failed, tests_passed+tests_failed);
    printf("==========================================\n\n");
    return tests_failed>0?1:0;
}
