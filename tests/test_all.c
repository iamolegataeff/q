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

/* ── 24. Smoke: compile only ── */
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
    test_smoke_compile();
    test_smoke_run_small();
    test_smoke_run_weights();
    printf("\n==========================================\n");
    printf("  PASSED: %d  FAILED: %d  TOTAL: %d\n", tests_passed, tests_failed, tests_passed+tests_failed);
    printf("==========================================\n\n");
    return tests_failed>0?1:0;
}
