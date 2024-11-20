
document.addEventListener('DOMContentLoaded', () => {
  
 
   const show = document.querySelector('#show');
   show.style.display='none';
   const form = document.querySelector('#inputform');
   form.onsubmit = function(e) {
      
      e.preventDefault(); // Prevent the default form submission
      const formData = new FormData(form);
      
      fetch("/grade", {
        method: "POST",
        body: formData,
        headers: {
          'X-CSRFToken': form.querySelector('[name=csrfmiddlewaretoken]').value,
        },
      })
      .then(response => response.json())
      .then(data => {
        console.log(data); // Handle the JSON response here
        // For example, update the DOM with the word count or feedback
        const newpage = document.createElement('div');
        console.log(typeof data[0]);
        
        newpage.innerHTML=`<div class="user-info" id=user-info>
        <h1>Result: </h1>
        <h1>${Math.floor(data['grade'])}</h1>
        `;
        show.style.display='block';
        show.replaceChildren("");
        show.append(newpage);
      })
      .catch(error => console.error('Error:', error));
    };
  });



