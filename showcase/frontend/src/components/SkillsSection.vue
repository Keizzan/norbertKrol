<script setup>
import technologies from '../data/technologies.json'
</script>

<template>
  <section id="skills" class="skills section">
    <div class="container section-title" data-aos="fade-up">
      <h2>Skills</h2>
    </div>

    <div class="container" data-aos="fade-up" data-aos-delay="100">
      <h3 class="text-center">Technologies</h3>

      <div class="row mb-5 ">
        <div
          v-for="technology in technologies"
          :key="technology.name"
          class="col-lg-3 col-6 mx-auto pb-3"
        >
          <div class="card h-100 mt-4">
            <div class="card-header text-center">
              <h5>{{ technology.name }}</h5>
            </div>

            <div class="card-body d-flex align-items-center justify-content-center">
              <img
                :src="technology.icon"
                :alt="`${technology.name} logo`"
                class="object-fit-contain img-fluid"
                height="80"
                width="80"
              />

            </div>
          </div>
        </div>
      </div>
    </div>
  </section>
</template>