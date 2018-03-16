//var dropzone = new Dropzone("#dz", { url: "upload.php"});
Dropzone.autoDiscover = false;

//var myDropzone = new Dropzone("#dz");

var myDropzone = new Dropzone("#dz", {
  url: "upload.php",
  addRemoveLinks: true,
  maxFilesize: 15, // MB
  parallelUploads: 15,
  autoProcessQueue: false,
  uploadMultiple: true,
  acceptedFiles: "image/*",
  maxFiles: 15,

  init: function() {
    var myDropzone = this;

    // First change the button to actually tell Dropzone to process the queue.
    document.querySelector("button[type=submit]").addEventListener("click", function(e) {
      // Make sure that the form isn't actually being sent.
      e.preventDefault();
      e.stopPropagation();
      myDropzone.processQueue();
    });


    this.on("sending", function(file, xhr, formData) {
      let name = document.querySelector("input[name=name]").value;
      formData.append("name", name);
    });

    this.on("success", function(file, response) {
      console.log(file);
      console.log(response);

      if(response.success) {
        window.location = "success.html";
      } else {

      }
    });

  }
  
});