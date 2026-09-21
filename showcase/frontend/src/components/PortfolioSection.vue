<script setup>
import projects from '../data/projects.json'
</script>

<template>
  <section id="portfolio" class="portfolio section">
    <div class="container section-title" data-aos="fade-up">
      <h2>Portfolio</h2>
      <p>A selection of projects I've worked on.</p>
    </div>

    <div class="container" data-aos="fade-up" data-aos-delay="100">
      <div class="row">
        <div v-for="project in projects" :key="project.name" class="col-lg-6 col-md-12 mb-4">
          <div class="card h-100">
            <div class="card-header">
              <h4>{{ project.name }}</h4>
            </div>

            <div class="card-body">
              <p>{{ project.description }}</p>
            </div>

            <div class="card-footer">
              <a :href="project.link" target="_blank">
                Open in new window
              </a>
            </div>
          </div>
        </div>
      </div>
    </div>
  </section>
</template>