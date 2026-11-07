/*
 * test_all.c — PostGPT-Q tests against the engine itself: postgpt_q.c is compiled in, its main() renamed.
 * Run from the repo root:  gcc tests/test_all.c -O2 -lm -o test_all && ./test_all
 * Engine runs (smoke, Q_SEED determinism, memory reload) happen in a temporary directory, never in the repo.
 */
#define main q_engine_main
#include "../postgpt_q.c"
#undef main
#include <sys/wait.h>
#include <fcntl.h>

static int tests_passed = 0;
static int tests_failed = 0;
static char ROOT[1024], TMPD[1024];

#define TEST(name) printf("  [TEST] %s... ", name);
#define PASS() do { printf("PASS\n"); tests_passed++; } while(0)
#define FAIL(msg) do { printf("FAIL: %s\n", msg); tests_failed++; } while(0)
#define CHECK(cond, msg) do { if(!(cond)) { FAIL(msg); return; } } while(0)

/* engine functions print progress lines; keep the test report readable */
static int quiet_fd=-1;
static void quiet(int on){
    fflush(stdout);
    if(on){ quiet_fd=dup(1); int n=open("/dev/null",O_WRONLY); dup2(n,1); close(n); }
    else if(quiet_fd>=0){ dup2(quiet_fd,1); close(quiet_fd); quiet_fd=-1; }
}
static int have_weights(void){ return access("weights/exported_weights.bin",F_OK)==0; }
static int load_weights(TF *t){ quiet(1); int ok=tf_load(t,"weights/exported_weights.bin"); quiet(0); return ok; }
static char *slurp(const char *path){
    FILE *f=fopen(path,"rb"); if(!f) return NULL;
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    char *b=malloc(n+1); n=(long)fread(b,1,n,f); b[n]=0; fclose(f); return b;
}
/* runs the compiled engine in dir/ with the given stdin; stdout+stderr go to dir/name.out; returns the exit code */
static int run_engine(const char *dir, const char *env, int weights, const char *input, const char *name){
    char cmd[4096];
    snprintf(cmd,sizeof(cmd),"mkdir -p '%s' && cd '%s' && ln -sfn '%s/docs' docs && printf '%s' | %s '%s/q' %s%s%s '%s/q.merges' '%s/tiny.txt' > '%s.out' 2>&1",
        dir,dir,ROOT,input,env,TMPD,weights?"'":"",weights?ROOT:"",weights?"/weights/exported_weights.bin'":"",ROOT,TMPD,name);
    int rc=system(cmd); return WIFEXITED(rc)?WEXITSTATUS(rc):-1;
}

/* ── 1. Math ── */
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

/* ── 3. MetaWeights ── */
void test_metaweights_bigram(void) {
    TEST("metaweights_bigram");
    int ids[]={0,1,0,1,0,1};
    MetaW *mw=calloc(1,sizeof(MetaW));
    quiet(1); meta_build(mw,ids,6,4); quiet(0);
    CHECK(mw->n_bi==2, "2 bigrams");
    CHECK(fabsf(meta_bi(mw,0,1)-1.0f)<1e-6f, "P(1|0)=1");
    CHECK(meta_bi(mw,1,1)<1e-9f, "unseen bigram gets the floor");
    free(mw);
    PASS();
}

void test_word_capture(void) {
    TEST("word_capture_ingest");
    MetaW *mw=calloc(1,sizeof(MetaW));
    int a[]={5,6}; ingest_ids(mw,a,2,0.02f);
    CHECK(mw->n_bi==1&&fabsf(mw->bigrams[0].prob-0.05f)<1e-6f, "new bigram starts at the 0.05 floor");
    ingest_ids(mw,a,2,0.02f);
    CHECK(mw->n_bi==1&&fabsf(mw->bigrams[0].prob-0.07f)<1e-6f, "existing bigram strengthened, not duplicated");
    int b[]={7,8}; ingest_ids(mw,b,2,0.02f);
    CHECK(mw->n_bi==2&&mw->bigrams[1].a==7&&mw->bigrams[1].b==8, "new bigram appended");
    free(mw);
    PASS();
}

/* ── 4. Chambers ── */
void test_chambers_init(void) {
    TEST("chambers_init");
    Chambers c; ch_init(&c);
    CHECK(fabsf(c.act[CH_LOVE]-0.2f)<1e-6f, "LOVE=0.2");
    CHECK(fabsf(c.act[CH_FLOW]-0.15f)<1e-6f, "FLOW=0.15");
    CHECK(c.act[CH_FEAR]==0, "FEAR=0"); CHECK(c.debt==0, "debt=0");
    PASS();
}

void test_chambers_decay(void) {
    TEST("chambers_decay");
    Chambers c; ch_init(&c);
    for(int i=0;i<6;i++) c.act[i]=0.5f;   /* equal phases: Kuramoto coupling sin(0)=0, only decay acts */
    ch_xfire(&c,1);
    for(int i=0;i<6;i++) CHECK(fabsf(c.act[i]-0.5f*CH_D[i])<1e-6f, "act decays by the engine's CH_D");
    PASS();
}

void test_chamber_modulation(void) {
    TEST("chamber_modulation");
    Chambers c; ch_init(&c);
    float act[6]={0.1f,0.6f,0.2f,0.1f,0.5f,0.3f}; memcpy(c.act,act,sizeof(act));
    float a,b,g,t; ch_modulate(&c,&a,&b,&g,&t);
    CHECK(a>1.3f&&a<1.4f, "alpha modulated");
    CHECK(b>1.1f&&b<1.3f, "beta modulated");
    CHECK(g>1.1f&&g<1.4f, "gamma modulated");
    CHECK(t>0.8f&&t<1.0f, "tau modulated");
    PASS();
}

void test_somatic_modulation(void) {
    TEST("somatic_modulation");
    Chambers c; ch_init(&c);
    float act[6]={0.1f,0.4f,0.1f,0.1f,0.5f,0.2f}, soma[6]={0.1f,0.8f,0.0f,0.0f,0.7f,0.3f};
    memcpy(c.act,act,sizeof(act)); memcpy(c.soma,soma,sizeof(soma)); c.presence=0.6f;
    float a,b,g,t; ch_modulate(&c,&a,&b,&g,&t);
    CHECK(a>1.5f, "love-flow soma boosts alpha");
    CHECK(b>1.2f, "flow-complex soma boosts beta");
    CHECK(g>1.2f, "presence boosts gamma");
    CHECK(t<1.0f, "flow cools tau");
    PASS();
}

void test_velocity_profile(void) {
    TEST("velocity_profile");
    Chambers c; ch_init(&c);
    VelocityProfile up=velocity_profile(&c,0.9f);
    CHECK(up.mode==VEL_UP, "high dissonance runs UP");
    CHECK(up.temp_mul>1.2f&&up.pro_mul>1.2f, "UP heats sampling and boosts prophecy");
    c.trauma=0.7f;
    VelocityProfile br=velocity_profile(&c,0.4f);
    CHECK(br.mode==VEL_BREATHE, "trauma breathes");
    CHECK(br.debt_decay<1.0f&&br.trauma_decay<1.0f, "BREATHE releases debt and trauma");
    PASS();
}

void test_periodic_mapping(void) {
    TEST("periodic_mapping");
    PeriodicTable *pt=calloc(1,sizeof(PeriodicTable)); periodic_init(pt);
    periodic_build_from_text(pt,"love rhythm paradox lumen love rhythm mystery lumen");
    int idx=periodic_find(pt,"lumen");
    CHECK(idx>=0, "unknown word classified from anchor neighbours");
    Chambers c; ch_init(&c); memset(c.act,0,sizeof(c.act));
    ch_feel_text(&c,"love rhythm paradox",pt);
    CHECK(c.act[CH_LOVE]>0&&c.act[CH_FLOW]>0&&c.act[CH_CMPLX]>0, "multiple chambers discoverable");
    free(pt);
    PASS();
}

/* ── 5. DOE Parliament ── */
void test_parliament_expert(void) {
    TEST("parliament_expert_forward");
    srand(1);
    Expert e; expert_init(&e,8,8,DOE_RANK);
    float x[8]={1,0,0,0,0,0,0,0}, out[8];
    expert_forward(&e,x,out);
    float sum=0; for(int i=0;i<8;i++) sum+=fabsf(out[i]);
    CHECK(sum>0, "nonzero"); CHECK(sum<1, "small");
    free(e.A); free(e.B); free(e.trace);
    PASS();
}

void test_parliament_vitality(void) {
    TEST("parliament_vitality");
    srand(2);
    Parliament p; parl_init(&p,8,4);
    float x[8]={0.5f,-0.2f,0.1f,0.9f,-0.4f,0.3f,0.0f,0.7f}, res[8];
    parl_election(&p,x,res);
    float mn=1; for(int i=0;i<p.n;i++){ CHECK(p.ex[i].vitality>=0&&p.ex[i].vitality<=1, "vitality in [0,1]"); if(p.ex[i].vitality<mn) mn=p.ex[i].vitality; }
    CHECK(mn<1.0f, "an election moves vitality");
    CHECK(p.last_k>=1&&p.last_k<=p.n&&p.n_winners>=1, "variable-k election picked winners");
    for(int i=0;i<p.n;i++){ free(p.ex[i].A); free(p.ex[i].B); free(p.ex[i].trace); }
    PASS();
}

/* ── 6. Transformer gate ── */
void test_transformer_gate(void) {
    TEST("transformer_gate");
    if(!have_weights()){printf("SKIP (no .bin weights)\n");tests_passed++;return;}
    TF t={0}; CHECK(load_weights(&t), "weights load");
    tf_forward(&t,72,0);
    float mag=0; for(int v=0;v<t.V;v++) mag+=fabsf(t.logits[v]); mag/=t.V;
    CHECK(mag>0.1f, "trained weights speak");
    for(int i=0;i<t.V*t.D;i++) t.tok[i]*=0.05f;   /* untrained-scale embeddings: logit magnitude falls under the gate */
    tf_reset(&t); tf_forward(&t,72,0);
    int nz=0; for(int v=0;v<t.V;v++) if(t.logits[v]!=0.0f) nz++;
    CHECK(nz==0, "untrained-scale logits are silenced to exactly zero");
    PASS();
}

/* ── 7. Boundary and sentence-start detection ── */
static int count_starts(const BPE *bpe, const char *text, int *first_byte){
    int ids[128]; int n=bpe_encode(bpe,(const uint8_t*)text,(int)strlen(text),ids,128), k=0;
    for(int p=1;p<n;p++) if(is_sentence_start(bpe,ids,p)){ if(!k) *first_byte=bpe->vocab_bytes[ids[p]][0]; k++; }
    return k;
}
void test_boundary(void) {
    TEST("boundary_detection");
    BPE bpe; bpe_load(&bpe,"q.merges");
    CHECK(is_boundary(&bpe,'.'), ". is boundary");
    CHECK(!is_boundary(&bpe,'a'), "a not boundary");
    int fb=0;
    CHECK(count_starts(&bpe,"It grows.\n\nWords that appear",&fb)==1&&fb=='W', "BPE-glued \".\\n\\n\" + capital opens a sentence");
    CHECK(count_starts(&bpe,"the sea. The sky is",&fb)==0, "a boundary token that swallowed the next word gives no start");
    PASS();
}

/* ── 8. Coherence scoring ── */
void test_coherence(void) {
    TEST("coherence_score");
    MetaW *mw=calloc(1,sizeof(MetaW));
    int ids[20]; for(int i=0;i<20;i++) ids[i]=i+1;
    CHECK(fabsf(coherence_score(mw,ids,5,1280)+0.5f)<1e-6f, "short penalty");
    CHECK(fabsf(coherence_score(mw,ids,20,1280)-1.5f)<1e-6f, "long bonus");
    free(mw);
    PASS();
}

/* ── 9. Sampling and prophecy ── */
void test_nucleus(void) {
    TEST("nucleus_sampling");
    float p[]={10,1,0.5f,0.1f,-5};
    srand(3);
    for(int k=0;k<200;k++) CHECK(sample_nucleus(p,5,1.0f,0.85f)==0, "peaked logits: nucleus is the top token");
    PASS();
}

void test_prophecy(void) {
    TEST("prophecy_query");
    MetaW *mw=calloc(1,sizeof(MetaW));
    BigramE bis[4]={{0,1,0.8f},{0,2,0.5f},{1,3,0.9f},{2,0,0.3f}};
    memcpy(mw->bigrams,bis,sizeof(bis)); mw->n_bi=4;
    int ctx[]={0,1}; float out[4];
    meta_prophecy(mw,ctx,2,out,4);
    CHECK(out[3]>0, "predicts token 3"); CHECK(out[2]>0, "predicts token 2");
    CHECK(out[0]==0, "skips appeared 0"); CHECK(out[1]==0, "skips appeared 1");
    free(mw);
    PASS();
}

void test_active_prophecy_state(void) {
    TEST("active_prophecy_state");
    MetaW *mw=calloc(1,sizeof(MetaW));
    prophecy_add(mw,42,0.7f);
    prophecy_update(mw,7);
    CHECK(mw->n_prophecy==1, "still present after unrelated token");
    CHECK(mw->prophecies[0].target==42&&mw->prophecies[0].age==1, "target kept, age incremented");
    CHECK(mw->prophecies[0].strength<0.7f, "strength decayed");
    prophecy_update(mw,42);
    CHECK(mw->n_prophecy==0, "fulfilled prophecy removed");
    free(mw);
    PASS();
}

void test_prophecy_pressure_ageing(void) {
    TEST("prophecy_pressure_ageing");
    MetaW *mw=calloc(1,sizeof(MetaW));
    prophecy_add(mw,7,0.6f);
    float fresh=prophecy_pressure(mw);
    for(int i=0;i<5;i++) prophecy_update(mw,99);
    float aged=prophecy_pressure(mw);
    CHECK(aged>fresh, "aged pressure exceeds fresh zero-age hint");
    CHECK(aged>0.0f, "aged pressure positive");
    free(mw);
    PASS();
}

/* ── 10. SPA ── */
void test_spa_embed(void) {
    TEST("spa_embedding");
    srand(42);
    SPACtx *s=calloc(1,sizeof(SPACtx)); spa_init(s,1280);
    int ids[]={5,9,77}; float e[SPA_DIM];
    spa_embed_sentence(s,ids,3,e);
    float norm=0; for(int d=0;d<SPA_DIM;d++) norm+=e[d]*e[d];
    CHECK(fabsf(norm-1.0f)<1e-3f, "unit-norm sentence embedding");
    free(s);
    PASS();
}

/* ── 11. Interference ── */
static void set_kw(char kw[][32], int *n, const char *a, const char *b, const char *c){
    *n=0; if(a) strcpy(kw[(*n)++],a); if(b) strcpy(kw[(*n)++],b); if(c) strcpy(kw[(*n)++],c);
}
void test_interference_doc_choice(void) {
    TEST("interference_doc_choice");
    Interference *itf=calloc(1,sizeof(Interference)); itf->n_docs=2;
    set_kw(itf->docs[0].keywords,&itf->docs[0].n_keywords,"fungus","mycelium","forest");
    set_kw(itf->docs[1].keywords,&itf->docs[1].n_keywords,"resonance","choir","counterpoint");
    itf->docs[0].n_heavy=itf->docs[1].n_heavy=2;
    srand(4);
    const InterferenceDoc *d=interf_choose_doc(itf,"resonance in the choir",NULL,NULL,NULL,NULL);
    CHECK(d==&itf->docs[1], "prompt resonates with matching doc");
    free(itf);
    PASS();
}

void test_chunk_resonance_choice(void) {
    TEST("chunk_resonance_choice");
    InterferenceDoc *doc=calloc(1,sizeof(InterferenceDoc)); doc->n_chunks=2;
    set_kw(doc->chunks[0].keywords,&doc->chunks[0].n_keywords,"fungus","forest",NULL);
    set_kw(doc->chunks[1].keywords,&doc->chunks[1].n_keywords,"choir","resonance",NULL);
    srand(5);
    const InterferenceChunk *c=interf_choose_chunk(doc,"resonance in the choir",NULL,NULL,NULL,NULL);
    CHECK(c==&doc->chunks[1], "matching chunk outranks unrelated chunk");
    free(doc);
    PASS();
}

void test_interference_seed(void) {
    TEST("interference_seed");
    BPE *bpe=calloc(1,sizeof(BPE)); bpe->vocab_size=2;
    memcpy(bpe->vocab_bytes[0],"Resonance",9); bpe->vocab_len[0]=9;   /* FLOW anchor */
    memcpy(bpe->vocab_bytes[1],"Void",4); bpe->vocab_len[1]=4;        /* VOID anchor */
    InterferenceChunk ck={0}; ck.n_heavy=2; ck.heavy[0]=1; ck.heavy[1]=0;
    Chambers c; ch_init(&c); memset(c.act,0,sizeof(c.act)); c.act[CH_FLOW]=0.8f;
    srand(6);
    CHECK(interf_seed_from_chunk(&ck,&c,bpe,NULL)==0, "dominant chamber prefers resonant seed");
    free(bpe);
    PASS();
}

/* ── 12. Memory (q.sqlite) ── */
void test_sqlite_memory_roundtrip(void) {
    TEST("sqlite_memory_reload_no_duplicates");
    char db[1200]; snprintf(db,sizeof(db),"%s/unit.sqlite",TMPD);
    int ids[]={5,6,7,5,6,8};
    MetaW *a=calloc(1,sizeof(MetaW)); ingest_ids(a,ids,6,0.05f);
    a->trigrams[0].prob=0.9f;
    PeriodicTable *pt=calloc(1,sizeof(PeriodicTable)); periodic_init(pt);
    Chambers ch; ch_init(&ch);
    CHECK(qsqlite_save(a,db,pt,&ch), "save ok");
    MetaW *b=calloc(1,sizeof(MetaW)); ingest_ids(b,ids,6,0.05f);   /* the same keys already built, like the corpus field */
    Chambers ch2; ch_init(&ch2);
    CHECK(qsqlite_load(b,db,pt,&ch2), "load ok");
    CHECK(b->n_bi==a->n_bi&&b->n_tri==a->n_tri&&b->n_hebb==a->n_hebb, "reload merges keys instead of duplicating them");
    CHECK(fabsf(b->trigrams[0].prob-0.9f)<1e-6f, "the stronger stored value wins");
    CHECK(qsqlite_load(b,db,pt,&ch2)&&b->n_tri==a->n_tri&&b->n_hebb==a->n_hebb, "a second reload adds nothing");
    free(a); free(b); free(pt);
    PASS();
}

/* ── 13. Destiny ── */
void test_destiny_steers_prompt_choice(void) {
    TEST("destiny_steers_prompt_choice");
    if(!have_weights()){printf("SKIP (no .bin weights)\n");tests_passed++;return;}
    BPE *bpe=calloc(1,sizeof(BPE)); bpe_load(bpe,"q.merges");
    char *txt=slurp("q.txt"); CHECK(txt!=NULL, "corpus");
    int n=(int)strlen(txt); if(n>20000) n=20000;
    int *cids=malloc(n*sizeof(int)), clen=bpe_encode(bpe,(const uint8_t*)txt,n,cids,n);
    int *starts=malloc(clen*sizeof(int)), ns=0;
    for(int p=1;p+4<clen;p++) if(is_sentence_start(bpe,cids,p)) starts[ns++]=p;
    CHECK(ns>=CHAIN_STEPS, "corpus has sentence starts to choose from");
    TF t={0}; CHECK(load_weights(&t), "weights load");
    float *zero=calloc(t.D,sizeof(float)), *ga=malloc(t.D*sizeof(float)), *gb=malloc(t.D*sizeof(float));
    int ta=cids[starts[0]], tb=-1;
    for(int i=1;i<ns&&tb<0;i++) if(cids[starts[i]]!=ta) tb=cids[starts[i]];
    for(int d=0;d<t.D;d++){ ga[d]=t.tok[ta*t.D+d]; gb[d]=t.tok[tb*t.D+d]; }
    srand(7); int s0=destiny_start(&t,cids,starts,ns,zero);
    srand(7); int sa=destiny_start(&t,cids,starts,ns,ga);
    srand(7); int sb=destiny_start(&t,cids,starts,ns,gb);
    CHECK(!(s0==sa&&sa==sb), "with weights, destiny changes which corpus sentence opens the step");
    memset(t.tok,0,(size_t)t.V*t.D*sizeof(float));   /* no weights: zero embeddings */
    srand(7); int z0=destiny_start(&t,cids,starts,ns,zero);
    srand(7); int za=destiny_start(&t,cids,starts,ns,ga);
    CHECK(z0==za&&z0==s0, "with zero embeddings destiny is silent: the draw is the ablated one");
    free(zero); free(ga); free(gb); free(starts); free(cids); free(txt); free(bpe);
    PASS();
}

/* ── 14. Engine runs (in TMPD) ── */
static int engine_built=0;
void test_smoke_compile(void) {
    TEST("smoke_compile");
    char cmd[3000];
    snprintf(cmd,sizeof(cmd),"cc '%s/postgpt_q.c' -O2 -lm -o '%s/q' 2>/dev/null && head -c 5000 '%s/q.txt' > '%s/tiny.txt'",ROOT,TMPD,ROOT,TMPD);
    CHECK(system(cmd)==0, "compiles");
    engine_built=1;
    PASS();
}

void test_smoke_run_small(void) {
    TEST("smoke_run_small_corpus");
    CHECK(engine_built, "engine built");
    char dir[1200]; snprintf(dir,sizeof(dir),"%s/smoke",TMPD);
    CHECK(run_engine(dir,"",0,"quit\\n","run")==0, "runs on 5KB corpus");
    PASS();
}

void test_smoke_run_weights(void) {
    TEST("smoke_run_with_weights");
    if(!have_weights()){printf("SKIP (no .bin weights)\n");tests_passed++;return;}
    CHECK(engine_built, "engine built");
    char dir[1200]; snprintf(dir,sizeof(dir),"%s/smokew",TMPD);
    CHECK(run_engine(dir,"",1,"quit\\n","run")==0, "runs with weights");
    PASS();
}

void test_legacy_memory_tail(void) {
    TEST("legacy_memory_tail_compat");
    CHECK(engine_built, "engine built");
    char dir[1200], path[1300]; snprintf(dir,sizeof(dir),"%s/legacy",TMPD); snprintf(path,sizeof(path),"%s/q.memory",dir);
    mkdir(dir,0755);
    FILE *f=fopen(path,"wb"); CHECK(f!=NULL, "write open");
    uint32_t magic=0x514D454D, npe=0; int nb=1,nt=0,nh=0,a=7,b=9; float p=0.25f;
    fwrite(&magic,4,1,f); fwrite(&nb,4,1,f); fwrite(&nt,4,1,f); fwrite(&nh,4,1,f);
    fwrite(&a,4,1,f); fwrite(&b,4,1,f); fwrite(&p,4,1,f); fwrite(&npe,4,1,f);   /* no somatic tail */
    fclose(f);
    CHECK(run_engine(dir,"Q_SEED=1",0,"quit\\n","run")==0, "engine runs on a legacy shard");
    char out[1400]; snprintf(out,sizeof(out),"%s/run.out",dir);
    char *s=slurp(out); CHECK(s&&strstr(s,"[memory loaded: 1 bi, 0 tri, 0 hebb from q.memory]"), "legacy shard loads without the somatic tail");
    free(s);
    PASS();
}

void test_engine_seed_determinism(void) {
    TEST("engine_q_seed_determinism");
    CHECK(engine_built, "engine built");
    for(int w=0;w<=have_weights();w++){
        char d1[1200],d2[1200],o1[1300],o2[1300];
        snprintf(d1,sizeof(d1),"%s/seed%dA",TMPD,w); snprintf(d2,sizeof(d2),"%s/seed%dB",TMPD,w);
        CHECK(run_engine(d1,"Q_SEED=42",w,"the forest remembers\\nquit\\n","run")==0&&run_engine(d2,"Q_SEED=42",w,"the forest remembers\\nquit\\n","run")==0, "both runs succeed");
        snprintf(o1,sizeof(o1),"%s/run.out",d1); snprintf(o2,sizeof(o2),"%s/run.out",d2);
        char *a=slurp(o1),*b=slurp(o2);
        CHECK(a&&b&&strstr(a,"[ 1]"), "chain printed");
        int same=strcmp(a,b)==0; free(a); free(b);
        CHECK(same, w?"Q_SEED=42 twice gives byte-identical output (weights)":"Q_SEED=42 twice gives byte-identical output (MetaWeights only)");
    }
    PASS();
}

void test_engine_memory_survives_reload(void) {
    TEST("engine_memory_survives_reload");
    CHECK(engine_built, "engine built");
    char dir[1200],o[1300]; snprintf(dir,sizeof(dir),"%s/mem",TMPD); snprintf(o,sizeof(o),"%s/run.out",dir);
    CHECK(run_engine(dir,"Q_SEED=5",0,"the forest remembers\\nquit\\n","run")==0, "session 1");
    char *s=slurp(o); int b1=-1,t1=-1,h1=-1; char *m=s?strstr(s,"[memory saved: "):NULL;
    CHECK(m&&sscanf(m,"[memory saved: %d bi, %d tri, %d hebb",&b1,&t1,&h1)==3, "session 1 saved");
    free(s);
    CHECK(run_engine(dir,"Q_SEED=6",0,"rain on a roof\\nquit\\n","run")==0, "session 2");
    s=slurp(o); int b2=-1,t2=-1,h2=-1; m=s?strstr(s,"[memory loaded: "):NULL;
    CHECK(m&&sscanf(m,"[memory loaded: %d bi, %d tri, %d hebb",&b2,&t2,&h2)==3, "session 2 loaded q.sqlite");
    free(s);
    CHECK(b2==b1&&t2==t1&&h2==h1, "session 2 starts from exactly what session 1 saved (no duplicates, nothing lost)");
    PASS();
}

/* the repo root must look the same after the suite: engine runs live in TMPD */
static struct stat root_before[3]; static int root_had[3];
static const char *ROOT_FILES[3]={"q.sqlite","q.memory","spores"};
static void snapshot_root(struct stat *st, int *had){ for(int i=0;i<3;i++) had[i]=stat(ROOT_FILES[i],&st[i])==0; }
void test_repo_root_untouched(void) {
    TEST("repo_root_untouched");
    struct stat after[3]; int had[3]; snapshot_root(after,had);
    for(int i=0;i<3;i++){
        CHECK(had[i]==root_had[i], "no q.sqlite / q.memory / spores appeared or vanished in the repo root");
        if(had[i]) CHECK(after[i].st_mtime==root_before[i].st_mtime&&after[i].st_size==root_before[i].st_size, "existing memory in the repo root untouched");
    }
    PASS();
}

int main(void) {
    printf("\n========== PostGPT-Q Test Suite ==========\n\n");
    if(!getcwd(ROOT,sizeof(ROOT))||access("postgpt_q.c",F_OK)!=0){ printf("run from the repo root\n"); return 1; }
    const char *tmp=getenv("TMPDIR"); snprintf(TMPD,sizeof(TMPD),"%s/q_tests_XXXXXX",tmp&&tmp[0]?tmp:"/tmp");
    if(!mkdtemp(TMPD)){ printf("mkdtemp failed\n"); return 1; }
    snapshot_root(root_before,root_had);
    test_clampf(); test_rmsnorm(); test_matmul(); test_softmax();
    test_bpe_load(); test_bpe_encode(); test_bpe_roundtrip();
    test_metaweights_bigram(); test_word_capture();
    test_chambers_init(); test_chambers_decay();
    test_chamber_modulation(); test_somatic_modulation(); test_velocity_profile(); test_periodic_mapping();
    test_parliament_expert(); test_parliament_vitality();
    test_transformer_gate();
    test_boundary();
    test_coherence();
    test_nucleus(); test_prophecy(); test_active_prophecy_state(); test_prophecy_pressure_ageing();
    test_spa_embed();
    test_interference_doc_choice(); test_chunk_resonance_choice(); test_interference_seed();
    test_sqlite_memory_roundtrip();
    test_destiny_steers_prompt_choice();
    test_smoke_compile(); test_smoke_run_small(); test_smoke_run_weights();
    test_legacy_memory_tail();
    test_engine_seed_determinism();
    test_engine_memory_survives_reload();
    test_repo_root_untouched();
    {char cmd[1200]; snprintf(cmd,sizeof(cmd),"rm -rf '%s'",TMPD); system(cmd);}
    printf("\n==========================================\n");
    printf("  PASSED: %d  FAILED: %d  TOTAL: %d\n", tests_passed, tests_failed, tests_passed+tests_failed);
    printf("==========================================\n\n");
    return tests_failed>0?1:0;
}
