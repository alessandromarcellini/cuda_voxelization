// 3 streams:
//  - 1 H2D
//  - 1 Kernel
//  - 1 D2H

// uso memoria host pinned per trasfermenti

// uso di cuda events per sincronizzare gli stream tra loro

// allocazione del numero massimo di punti (stimato se fosse un caso reale) per ospitare i punti ogni tot su device

// più buffer per ospitare i punti sul device, non un solo buffer con offset (è più complesso da gestire e non cambia praticamente niente)
// uso di cuda events per segnalare quando un buffer è stato elaborato e quindi può essere sovrascritto (un evento che segnala che il buffer è libero)
// uso di ringbuffer per gestire l'uso dei buffer


#define NUM_BUFFERS 5





// xxxxxxxxxxxxxxxxxxxxxxxx // evento sovrascrivibile
// xxxxxxxxxxxxxxxxxxxxxxxx
// xxxxxxxxxxxxxxxxxxxxxxxx
// ------------------------
// ------------------------